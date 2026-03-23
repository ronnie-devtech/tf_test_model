#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一日志管理工具

为所有模型推理脚本提供统一的日志输出接口，支持：
- 统一的日志格式和目录结构
- 按模型名称自动隔离日志
- 同时输出到文件和终端
- 性能结果的统一保存
- 未来模型的无缝扩展
"""

import os
import sys
import glob
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ==========================================
# 全局注册表：已创建的 LogManager 实例
# 通过模型名做 key，避免同一模型重复创建
# ==========================================
_LOG_MANAGERS: Dict[str, "LogManager"] = {}


# ==========================================
# 默认配置
# ==========================================
DEFAULT_LOG_ROOT = "logs"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL = logging.INFO


def get_musa_plugin_path_candidates() -> List[str]:
    """Return candidate libmusa_plugin.so paths.

    Keep the workspace-relative path as the preferred default because it works
    for both a local clone and a typical sibling checkout layout. The docker
    absolute path is still retained as an explicit fallback for container runs.
    """
    repo_root = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(repo_root)

    candidate_paths = []

    env_path = os.environ.get("MUSA_PLUGIN_PATH")
    if env_path:
        candidate_paths.append(env_path)

    candidate_paths.extend(
        [
            os.path.join(
                workspace_root,
                "tensorflow_musa_extension",
                "build",
                "libmusa_plugin.so",
            ),
            os.path.join(
                workspace_root,
                "tensorflow_musa_extension",
                "build_local",
                "libmusa_plugin.so",
            ),
            # Docker absolute fallback.
            "/workspace/tensorflow_musa_extension/build/libmusa_plugin.so",
        ]
    )

    return candidate_paths


def get_default_musa_plugin_path() -> str:
    """Return the best default path for libmusa_plugin.so."""
    candidate_paths = get_musa_plugin_path_candidates()

    for candidate in candidate_paths:
        normalized = os.path.abspath(os.path.expanduser(candidate))
        if os.path.exists(normalized):
            return normalized

    return os.path.abspath(os.path.expanduser(candidate_paths[0]))


def resolve_musa_plugin_path(plugin_path: Optional[str]) -> str:
    """Normalize a user-provided plugin path, or auto-detect one."""
    if plugin_path:
        return os.path.abspath(os.path.expanduser(plugin_path))
    return get_default_musa_plugin_path()


def load_latest_after_fusion_graph_def(
    logger: Optional[logging.Logger] = None,
    warn_if_missing: bool = True,
) -> Tuple[Optional[Any], Optional[str]]:
    """Load the newest dumped `*_after_fusion.pbtxt` GraphDef if available."""
    dump_enabled = os.environ.get("MUSA_DUMP_GRAPHDEF", "")
    if dump_enabled not in ("1", "true", "TRUE", "yes"):
        return None, None

    dump_dir = os.environ.get("MUSA_DUMP_GRAPHDEF_DIR", ".")
    dump_files = sorted(glob.glob(os.path.join(dump_dir, "*_after_fusion.pbtxt")))
    if not dump_files:
        if logger is not None and warn_if_missing:
            logger.warning("No after_fusion dump found in %s", dump_dir)
        return None, None

    from google.protobuf import text_format
    from tensorflow.core.framework import graph_pb2

    latest_dump = dump_files[-1]
    graph_def = graph_pb2.GraphDef()
    with open(latest_dump, "r", encoding="utf-8") as handle:
        text_format.Parse(handle.read(), graph_def)

    return graph_def, latest_dump


def build_optimized_op_type_map(
    logger: Optional[logging.Logger] = None,
    warn_if_missing: bool = True,
) -> Tuple[Dict[str, str], Optional[str]]:
    """Build a node_name -> op_type map from the latest optimized graph dump."""
    graph_def, latest_dump = load_latest_after_fusion_graph_def(
        logger=logger, warn_if_missing=warn_if_missing
    )
    if graph_def is None:
        return {}, None

    return {node.name: node.op for node in graph_def.node}, latest_dump


class LogManager:
    """统一日志管理器

    每个模型实例化一个 LogManager，自动管理：
    - trace 目录的创建
    - logger 的创建和 handler 绑定
    - 性能结果 JSON 的保存

    使用示例:
        # 在推理脚本中
        from common.utils import get_log_manager

        log_mgr = get_log_manager("wukong_inference")
        logger = log_mgr.get_logger("profiler")
        logger.info("开始推理...")

        # 保存性能结果
        log_mgr.save_json("performance_result.json", perf_dict)
    """

    def __init__(
        self,
        model_name: str,
        log_root: str = DEFAULT_LOG_ROOT,
        log_level: int = DEFAULT_LOG_LEVEL,
        log_format: str = DEFAULT_LOG_FORMAT,
        date_format: str = DEFAULT_DATE_FORMAT,
        enable_stdout: bool = True,
        enable_file: bool = True,
    ):
        """
        Args:
            model_name:    模型/场景名称，用于隔离日志目录
                           例如 "wukong_inference", "graph_inference"
            log_root:      日志根目录，默认 "logs"
            log_level:     日志级别，默认 INFO
            log_format:    日志格式
            date_format:   时间戳格式
            enable_stdout: 是否同时输出到终端
            enable_file:   是否写入日志文件
        """
        self.model_name = model_name
        self.log_root = log_root
        self.log_level = log_level
        self.log_format = log_format
        self.date_format = date_format
        self.enable_stdout = enable_stdout
        self.enable_file = enable_file

        # 已创建的 logger 缓存，避免重复添加 handler
        self._loggers: Dict[str, logging.Logger] = {}

        # 创建 trace 目录
        self._timestamp = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
        self._log_dir = os.path.join(self.log_root, self.model_name)
        self.trace_dir = os.path.join(self._log_dir, f"{self._timestamp}_trace")
        os.makedirs(self.trace_dir, exist_ok=True)

        # 共享 formatter
        self._formatter = logging.Formatter(
            fmt=self.log_format, datefmt=self.date_format
        )

    # --------------------------------------------------
    # 核心 API
    # --------------------------------------------------

    def get_logger(
        self, component: str = "main", log_file: Optional[str] = None
    ) -> logging.Logger:
        """获取一个已配置好的 logger

        同一 (model_name, component) 组合只会创建一次 logger，
        后续调用直接返回缓存的实例。

        Args:
            component:  组件名称，例如 "main", "profiler", "data_loader"
                        最终 logger 名称为 "{model_name}.{component}"
            log_file:   日志文件名（不含路径），默认为 "{component}.log"

        Returns:
            配置好的 logging.Logger 实例
        """
        logger_name = f"{self.model_name}.{component}"

        # 已创建则直接返回
        if logger_name in self._loggers:
            return self._loggers[logger_name]

        logger = logging.getLogger(logger_name)
        logger.setLevel(self.log_level)

        # 防止日志传播到 root logger 导致重复输出
        logger.propagate = False

        # 如果 logger 已有 handler（例如其他地方创建过），跳过
        if not logger.handlers:
            if self.enable_file:
                file_name = log_file or f"{component}.log"
                file_path = os.path.join(self.trace_dir, file_name)
                fh = logging.FileHandler(file_path, mode="a", encoding="utf-8")
                fh.setLevel(self.log_level)
                fh.setFormatter(self._formatter)
                logger.addHandler(fh)

            if self.enable_stdout:
                sh = logging.StreamHandler(sys.stdout)
                sh.setLevel(self.log_level)
                sh.setFormatter(self._formatter)
                logger.addHandler(sh)

        self._loggers[logger_name] = logger
        return logger

    def save_json(
        self, filename: str, data: Dict[str, Any], sub_dir: Optional[str] = None
    ) -> str:
        """将字典数据保存为 JSON 文件到 trace 目录

        Args:
            filename:  文件名，例如 "performance_result.json"
            data:      要保存的字典数据
            sub_dir:   可选子目录名，会在 trace_dir 下创建

        Returns:
            保存的完整文件路径
        """
        if sub_dir:
            save_dir = os.path.join(self.trace_dir, sub_dir)
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = self.trace_dir

        file_path = os.path.join(save_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return file_path

    def make_sub_dir(self, name: str) -> str:
        """在 trace_dir 下创建并返回子目录路径

        Args:
            name: 子目录名称

        Returns:
            子目录完整路径
        """
        sub_dir = os.path.join(self.trace_dir, name)
        os.makedirs(sub_dir, exist_ok=True)
        return sub_dir

    def get_timestamped_filename(self, prefix: str, suffix: str = ".json") -> str:
        """生成带时间戳的文件名

        Args:
            prefix: 文件名前缀
            suffix: 文件扩展名

        Returns:
            格式如 "prefix_MUSA_2026-03-04-12.30.00.json"
        """
        ts = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
        return f"{prefix}_{ts}{suffix}"

    # --------------------------------------------------
    # 便捷属性
    # --------------------------------------------------

    @property
    def timestamp(self) -> str:
        """返回本次运行的时间戳"""
        return self._timestamp

    def __repr__(self) -> str:
        return (
            f"LogManager(model_name={self.model_name!r}, "
            f"trace_dir={self.trace_dir!r})"
        )


# ==========================================
# 模块级工厂函数（推荐使用入口）
# ==========================================


def get_log_manager(
    model_name: str,
    log_root: str = DEFAULT_LOG_ROOT,
    log_level: int = DEFAULT_LOG_LEVEL,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    enable_stdout: bool = True,
    enable_file: bool = True,
) -> LogManager:
    """获取或创建指定模型的 LogManager 单例

    同一 model_name 只会创建一次 LogManager。
    这是推荐的使用入口。

    Args:
        model_name: 模型/场景名称
        其余参数:   仅在首次创建时生效

    Returns:
        LogManager 实例

    示例:
        # wukong 推理脚本
        log_mgr = get_log_manager("wukong_inference")

        # graph_def 推理脚本
        log_mgr = get_log_manager("graph_inference")

        # 未来新增的模型
        log_mgr = get_log_manager("new_model_inference")
    """
    if model_name not in _LOG_MANAGERS:
        _LOG_MANAGERS[model_name] = LogManager(
            model_name=model_name,
            log_root=log_root,
            log_level=log_level,
            log_format=log_format,
            date_format=date_format,
            enable_stdout=enable_stdout,
            enable_file=enable_file,
        )
    return _LOG_MANAGERS[model_name]


def reset_log_managers():
    """重置所有已注册的 LogManager（主要用于测试）"""
    _LOG_MANAGERS.clear()
