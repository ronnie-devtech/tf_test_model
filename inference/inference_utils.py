"""
公共测试工具函数

所有 inference 模型共用的工具函数。
"""

import argparse
import os
import logging
import sys
import random
import numpy as np
from pathlib import Path
import tensorflow as tf
from typing import Dict, List, Any, Optional

def setup_logger(model_name: str = "TF MUSA Inference"):
    """设置日志记录器

    Args:
        model_name: 模型名称，用于logger名称和日志文件名
    """
    # 1. 使用传入的模型名称创建logger
    logger = logging.getLogger(f"[{model_name}]")
    logger.setLevel(logging.DEBUG)  # Logger 级别设最低，让 Handler 去过滤

    # 2. 🛡️ 核心防护：防止多次 import 导致重复添加 Handler（日志会重复输出）
    if logger.hasHandlers():
        return logger

    # 3. 定义格式
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 4. 控制台 Handler
    console_h = logging.StreamHandler(sys.stdout)
    console_h.setLevel(logging.INFO)
    console_h.setFormatter(formatter)

    # 5. 文件 Handler（带轮转，避免日志撑爆磁盘）
    from logging.handlers import RotatingFileHandler
    # 使用模型名称作为日志文件名
    log_file = Path(f"logs/{model_name}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_h = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
    )
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(formatter)

    # 6. 绑定 Handler
    logger.addHandler(console_h)
    logger.addHandler(file_h)
    return logger

def setup_environment():
    """设置环境变量"""
    musa_visible_devices = os.environ.get("MUSA_VISIBLE_DEVICES", "")
    if musa_visible_devices:
        os.environ["MUSA_VISIBLE_DEVICES"] = musa_visible_devices
        print(f"MUSA_VISIBLE_DEVICES set to: {musa_visible_devices}")

def create_parent_parser():
    """创建父参数解析器，供子模块扩展使用"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "musa"],
        help="inference device: cpu, cuda, musa (default: cpu)",
    )
    parser.add_argument(
        "--xla",
        action="store_true",
        help="enable XLA JIT compilation (only applicable for CUDA device)",
    )
    parser.add_argument(
        "--musa-plugin",
        type=str,
        nargs="?",
        help="Path to the TensorFlow MUSA library .so file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--infer-iters",
        type=int,
        default=2000,
        help="Number of inference iterations (default: 2000)",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    return parser

def parse_arguments(parent_parser: Optional[argparse.ArgumentParser] = None):
    """解析命令行参数

    Args:
        parent_parser: 可选的父解析器，用于子模块添加自己的参数
    """
    base_parser = create_parent_parser()

    if parent_parser is not None:
        # 如果提供了父解析器，将其与基础解析器合并
        parser = argparse.ArgumentParser(
            description="Test TensorFlow MUSA Library",
            parents=[base_parser, parent_parser]
        )
    else:
        parser = argparse.ArgumentParser(
            description="Test TensorFlow MUSA Library",
            parents=[base_parser]
        )

    args = parser.parse_args()
    return args

def load_musa_plugin(plugin_path):
    """加载 TensorFlow MUSA 插件

    Args:
        plugin_path: 插件 .so 文件的路径

    Raises:
        FileNotFoundError: 如果路径为 None 或文件不存在
    """
    if plugin_path is None:
        raise FileNotFoundError(
            "MUSA plugin path not specified and could not be auto-detected.\n"
            "Please provide --musa_plugin argument or set MUSA_PLUGIN_PATH environment variable.\n"
            "Example: python test_tf_musa_extension.py --musa_plugin /path/to/libmusa_plugin.so"
        )
    if not os.path.exists(plugin_path):
        raise FileNotFoundError(
            f"MUSA plugin not found at: {plugin_path}\n"
            "Please build the plugin first:\n"
            "  cd tensorflow_musa_extension && ./build.sh"
        )
    tf.load_library(plugin_path)
    print(f"Successfully loaded tensorflow musa library from {plugin_path}")

def set_random_seeds(seed=42):
    """设置随机种子"""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def create_session_config(
    device_type: str = "cpu",
    xla: bool = False,
    log_device_placement: bool = False,
    allow_soft_placement: bool = False,
    logger: Optional[logging.Logger] = None,
):
    tf_version = int(tf.__version__.split('.')[0])

    if tf_version == 1:
        ConfigProto = tf.ConfigProto
        OptimizerOptions = tf.OptimizerOptions
    else:  # TF2+
        ConfigProto = tf.compat.v1.ConfigProto
        OptimizerOptions = tf.compat.v1.OptimizerOptions
        if logger:
            logger.debug("Using tf.compat.v1.ConfigProto for TF2.x")

    config = ConfigProto()
    config.allow_soft_placement = allow_soft_placement
    config.log_device_placement = log_device_placement

    device_type_upper = (device_type or "cpu").upper()

    if device_type_upper == "CUDA":
        config.gpu_options.allow_growth = True

        # XLA JIT 配置（注意：TF2 中 ON_1 可能需通过 compat 访问）
        if xla:
            try:
                config.graph_options.optimizer_options.global_jit_level = (
                    OptimizerOptions.ON_1
                )
                if logger:
                    logger.info("Enabled XLA JIT compilation for CUDA")
            except AttributeError:
                if logger:
                    logger.warning("XLA config path not found, skipping JIT setup")

    elif device_type_upper == "MUSA":
        rewrite_options = config.graph_options.rewrite_options
        custom_opt = rewrite_options.custom_optimizers.add()
        custom_opt.name = "musa_graph_optimizer"

        # 可选：传递 MUSA 特有参数（根据实际 SDK 文档调整）
        # custom_opt.parameter_map["some_key"].s = b"some_value"

        if logger:
            logger.info("Enabled custom optimizer: musa_graph_optimizer")

        # XLA 在 MUSA 上可能不支持或需特殊配置
        if xla and logger:
            logger.warning("XLA support on MUSA is experimental, proceed with caution")

    # ---------- CPU 配置（可选优化）----------
    elif device_type_upper == "CPU":
        # 可在此设置 CPU 线程数等（TF1/TF2 兼容）
        # config.intra_op_parallelism_threads = 4
        # config.inter_op_parallelism_threads = 2
        pass

    else:
        if logger:
            logger.warning(f"Unknown device_type: {device_type}, using default config")

    return config

def print_perf_result(batches, run_times, num_runs, logger: Optional[logging.Logger] = None):
    """打印性能统计结果

    Args:
        batches: batch数量
        run_times: 运行时间列表
        num_runs: 运行次数
        logger: 可选的日志记录器
    """
    # 性能统计
    total_time = sum(run_times)
    avg_time = total_time / num_runs
    min_time = min(run_times)
    max_time = max(run_times)
    p50 = np.percentile(run_times, 50)
    p95 = np.percentile(run_times, 95)
    p99 = np.percentile(run_times, 99)

    log = logger.info if logger else print
    log("\n" + "="*50)
    log("[性能统计]")
    log("="*50)
    log(f"  运行次数: {num_runs}")
    log(f"  总耗时:   {total_time:.2f} ms")
    log(f"  平均:     {avg_time:.4f} ms")
    log(f"  最小:     {min_time:.4f} ms")
    log(f"  最大:     {max_time:.4f} ms")
    log(f"  P50:      {p50:.4f} ms")
    log(f"  P95:      {p95:.4f} ms")
    log(f"  P99:      {p99:.4f} ms")
    log(f"  吞吐量:   {1000/avg_time * batches:.2f} samples/s")
    log("="*50)


def compare_accuracy(
    cpu_result: np.ndarray,
    device_result: np.ndarray,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    cossim: float = 0.9999,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """比较CPU结果与其他设备结果的精度

    Args:
        cpu_result: CPU计算结果
        device_result: 其他设备计算结果（如MUSA）
        atol: 绝对误差阈值，默认1e-5
        rtol: 相对误差阈值，默认1e-5
        cossim: 余弦相似度阈值，默认0.9999
        logger: 可选的日志记录器

    Returns:
        包含比对结果的字典，格式:
        {
            'passed': bool,  # 是否通过所有检查
            'atol_passed': bool,  # 绝对误差是否通过
            'rtol_passed': bool,  # 相对误差是否通过
            'cossim_passed': bool,  # 余弦相似度是否通过
            'max_absolute_error': float,  # 最大绝对误差
            'max_relative_error': float,  # 最大相对误差
            'cosine_similarity': float,  # 余弦相似度
            'mismatched_elements': int,  # 不匹配元素数量
            'total_elements': int,  # 总元素数量
        }
    """
    log = logger.info if logger else print

    # 确保输入是numpy数组
    cpu_result = np.asarray(cpu_result)
    device_result = np.asarray(device_result)

    # 检查形状是否一致
    if cpu_result.shape != device_result.shape:
        raise ValueError(
            f"Shape mismatch: CPU result shape {cpu_result.shape} vs "
            f"Device result shape {device_result.shape}"
        )

    # 检查数据类型
    if cpu_result.dtype != device_result.dtype:
        log(f"Warning: Dtype mismatch: CPU {cpu_result.dtype} vs Device {device_result.dtype}")

    # 计算绝对误差和相对误差
    abs_diff = np.abs(cpu_result - device_result)

    # 避免除以0的情况
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = abs_diff / (np.abs(cpu_result) + 1e-10)

    max_absolute_error = np.max(abs_diff)
    max_relative_error = np.max(rel_diff)

    # 计算余弦相似度
    cpu_flat = cpu_result.flatten()
    device_flat = device_result.flatten()

    # 处理全0的情况
    cpu_norm = np.linalg.norm(cpu_flat)
    device_norm = np.linalg.norm(device_flat)

    if cpu_norm == 0 and device_norm == 0:
        cosine_similarity = 1.0  # 两个都是0向量，认为完全相似
    elif cpu_norm == 0 or device_norm == 0:
        cosine_similarity = 0.0  # 一个是0向量，另一个不是
    else:
        cosine_similarity = np.dot(cpu_flat, device_flat) / (cpu_norm * device_norm)

    # 检查是否通过阈值
    atol_passed = max_absolute_error <= atol
    rtol_passed = max_relative_error <= rtol
    cossim_passed = cosine_similarity >= cossim

    # 计算不匹配元素数量（使用 atol 和 rtol 的组合判断）
    tolerance = atol + rtol * np.abs(cpu_result)
    mismatched = abs_diff > tolerance
    mismatched_elements = np.count_nonzero(mismatched)
    total_elements = cpu_result.size

    passed = atol_passed and rtol_passed and cossim_passed

    result = {
        'passed': passed,
        'atol_passed': atol_passed,
        'rtol_passed': rtol_passed,
        'cossim_passed': cossim_passed,
        'max_absolute_error': float(max_absolute_error),
        'max_relative_error': float(max_relative_error),
        'cosine_similarity': float(cosine_similarity),
        'mismatched_elements': int(mismatched_elements),
        'total_elements': int(total_elements),
    }

    # 打印比对结果
    log("[精度比对结果]")
    log("="*50)
    status = "PASSED" if passed else "FAILED"
    log(f"  整体状态:        {status}")
    log("-"*50)
    log(f"  绝对误差检查:    {'PASSED' if atol_passed else 'FAILED'} (max={max_absolute_error:.2e}, threshold={atol:.2e})")
    log(f"  相对误差检查:    {'PASSED' if rtol_passed else 'FAILED'} (max={max_relative_error:.2e}, threshold={rtol:.2e})")
    log(f"  余弦相似度检查:  {'PASSED' if cossim_passed else 'FAILED'} (value={cosine_similarity:.6f}, threshold={cossim})")
    log("-"*50)
    log(f"  不匹配元素:      {mismatched_elements}/{total_elements} ({100*mismatched_elements/total_elements:.4f}%)")
    log("="*50)

    return result


def check_acc(func, *args, atol=1e-5, rtol=1e-5, cossim=0.9999,
              logger: Optional[logging.Logger] = None):

    log = logger.info if logger else print