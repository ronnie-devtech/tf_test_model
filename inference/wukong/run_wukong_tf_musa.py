"""
Wukong 模型 TensorFlow 推理脚本
支持前向推理、性能分析、trace 文件生成和设备信息查看
支持 CPU/MUSA 设备性能对比
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from datetime import datetime
from typing import Tuple, List, Dict, Any
import threading
import collections

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tf_test_model.utils import (
    build_optimized_op_type_map,
    get_default_musa_plugin_path,
    get_log_manager,
    resolve_musa_plugin_path,
)

try:
    import tensorflow as tf
    import numpy as np

    TF_AVAILABLE = True
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
    print("Please ensure TensorFlow is installed and available in your environment.")
    TF_AVAILABLE = False

# 设置随机种子以确保可重现性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 只有在 TensorFlow 可用时才导入模型
if TF_AVAILABLE:
    from model.wukong import Wukong


class InferenceProfiler:
    """Wukong模型推理分析器"""

    # Eager profiler emits both real kernel events and a large amount of Python /
    # dispatcher overhead. Filter wrapper events so the final table is closer to
    # "actual operator time" instead of "eager runtime overhead time".
    _FILTERED_PROFILE_PREFIXES = (
        "TFE_Py_",
        "EagerKernelExecute",
        "convert_to_tensor",
        "ValidateInputTypeAndPlacement",
        "tf.constant",
        "EagerExecute: ",
        "EagerLocalExecute: ",
    )

    _FILTERED_PROFILE_NAMES = {
        "_SOURCE",
    }

    def __init__(self, batch_size: int = 1024, device_type: str = "MUSA"):
        if not TF_AVAILABLE:
            raise RuntimeError(
                "TensorFlow is not available. Please install TensorFlow to use this script."
            )

        self.batch_size = batch_size
        self.device_type = device_type.upper()
        self.log_mgr = get_log_manager("wukong_inference")
        self.logger = self.log_mgr.get_logger("profiler", "inference.log")
        self.trace_dir = self.log_mgr.trace_dir
        self.setup_model_and_data()
        self.profile_ops = False  # Flag to enable operator profiling
        self.operator_timings = collections.defaultdict(
            list
        )  # Store operator timing data

    def setup_model_and_data(self):
        """设置模型和测试数据"""
        # 模型配置
        self.num_cat_features = 26
        self.num_dense_features = 13
        self.num_sparse_embs = [
            1460,
            583,
            10131227,
            2202608,
            305,
            24,
            12517,
            633,
            3,
            93145,
            5683,
            8351593,
            3194,
            27,
            14992,
            5461306,
            10,
            5652,
            2173,
            4,
            7046547,
            18,
            15,
            286181,
            105,
            142572,
        ]

        # 创建模型
        self.model = Wukong(
            num_layers=8,
            num_sparse_embs=self.num_sparse_embs,
            dim_emb=128,
            dim_input_sparse=self.num_cat_features,
            dim_input_dense=self.num_dense_features,
            num_emb_lcb=32,
            num_emb_fmb=32,
            rank_fmb=24,
            num_hidden_wukong=3,
            dim_hidden_wukong=2048,
            num_hidden_head=2,
            dim_hidden_head=256,
            dim_output=1,
            dropout=0.5,
            bias=False,
        )

        # 构建模型（触发参数创建）
        dummy_sparse = tf.zeros((1, self.num_cat_features), dtype=tf.int32)
        dummy_dense = tf.zeros((1, self.num_dense_features), dtype=tf.float32)
        _ = self.model((dummy_sparse, dummy_dense))

        # 创建有效的测试数据
        self.test_sparse, self.test_dense = self._create_valid_test_data(
            self.batch_size
        )

        self.logger.info(
            f"Model created with {len(self.model.trainable_variables)} trainable variables"
        )
        self.logger.info(
            f"Test data shapes - Sparse: {self.test_sparse.shape}, Dense: {self.test_dense.shape}"
        )

    def _create_valid_test_data(self, batch_size: int):
        """创建有效的测试数据，确保稀疏特征索引不越界"""
        # 为每个稀疏特征列生成有效的索引
        sparse_data = []
        for emb_size in self.num_sparse_embs:
            # 生成 [0, emb_size) 范围内的随机整数
            indices = tf.random.uniform(
                (batch_size,), minval=0, maxval=emb_size, dtype=tf.int32
            )
            sparse_data.append(indices)

        # 转换为 [batch_size, num_cat_features] 的张量
        test_sparse = tf.stack(sparse_data, axis=1)
        test_dense = tf.random.normal((batch_size, self.num_dense_features))

        return test_sparse, test_dense

    def get_op_device_info(self):
        """获取算子设备信息"""
        device_info = {}

        # 获取当前可用设备
        devices = tf.config.list_physical_devices()
        self.logger.info(f"Available devices: {[d.name for d in devices]}")

        # 检查是否有MUSA设备
        musa_devices = tf.config.list_physical_devices("MUSA")
        if not musa_devices:
            # raise RuntimeError("未检测到 MUSA 设备，请检查驱动和 TensorFlow-MUSA 安装")
            print("No MUSA devices found, falling back to CPU")

        if musa_devices:
            self.logger.info(f"MUSA devices available: {len(musa_devices)}")
            for i, device in enumerate(musa_devices):
                self.logger.info(f"  MUSA device {i}: {device}")
        else:
            self.logger.info("No GPU/MUSA devices available, using CPU")

        # 尝试在不同设备上运行以确定实际使用的设备
        try:
            if musa_devices:
                with tf.device("/device:MUSA:0"):
                    result = self.model(
                        (self.test_sparse, self.test_dense), training=False
                    )
                    self.logger.info("Model successfully ran on MUSA device")
                    return "MUSA"
            else:
                with tf.device("/CPU:0"):
                    result = self.model(
                        (self.test_sparse, self.test_dense), training=False
                    )
                    self.logger.info("Model running on CPU")
                    return "CPU"
        except Exception as e:
            self.logger.warning(f"Failed to run on preferred device: {e}")
            # 回退到CPU
            with tf.device("/CPU:0"):
                result = self.model((self.test_sparse, self.test_dense), training=False)
                self.logger.info("Model running on CPU (fallback)")
                return "CPU"

    def profile_operator_times(self, inputs):
        """使用TensorFlow profiler分析算子运行时间"""
        # 启动profiler
        log_dir = f"{self.trace_dir}/ops_profile"
        os.makedirs(log_dir, exist_ok=True)

        # 预热运行，确保所有算子都被初始化
        _ = self.model(inputs, training=False)
        if hasattr(tf, "musa") and tf.config.list_physical_devices("/device:MUSA:0"):
            tf.musa.synchronize()

        # 开始性能分析 - use a specific profiler options
        options = tf.profiler.experimental.ProfilerOptions(
            host_tracer_level=2, python_tracer_level=0, device_tracer_level=1
        )

        tf.profiler.experimental.start(log_dir, options=options)

        # 执行推理
        start_time = time.time()
        result = self.model(inputs, training=False)
        end_time = time.time()

        # 确保设备同步
        if hasattr(tf, "musa") and tf.config.list_physical_devices("/device:MUSA:0"):
            tf.musa.synchronize()

        # 停止性能分析
        tf.profiler.experimental.stop()

        # 解析性能分析结果
        self.parse_operator_timings(log_dir)

        return result, end_time - start_time

    def parse_operator_timings(self, log_dir):
        """解析算子时间信息"""
        import gzip

        # 查找生成的trace文件 - look for common profiler output file names
        trace_files = []
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.endswith(".json") and (
                    "trace" in file.lower() or "profiler" in file.lower()
                ):
                    trace_files.append(os.path.join(root, file))

        # Look for compressed trace files (.json.gz)
        if not trace_files:
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                    if file.endswith(".json.gz") and "trace" in file.lower():
                        trace_files.append(os.path.join(root, file))

        # If no files found with 'trace' or 'profiler' in name, try any .json file
        if not trace_files:
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                    if file.endswith(".json"):
                        trace_files.append(os.path.join(root, file))

        # Also look for any .json.gz files if no .json files found
        if not trace_files:
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                    if file.endswith(".json.gz"):
                        trace_files.append(os.path.join(root, file))

        if not trace_files:
            self.logger.warning("No trace files found for operator timing analysis")
            self.logger.info(f"Looking in directory: {log_dir}")
            # List all files in the directory for debugging
            all_files = []
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                    all_files.append(os.path.join(root, file))
            if all_files:
                self.logger.info(f"Files found in directory: {all_files}")
            else:
                self.logger.info("No files found in the profiler output directory")
            return

        # 解析第一个trace文件
        trace_file = trace_files[0]
        try:
            if trace_file.endswith(".gz"):
                # Handle compressed file
                import gzip

                with gzip.open(trace_file, "rt", encoding="utf-8") as f:
                    trace_data = json.load(f)
            else:
                # Handle regular JSON file
                with open(trace_file, "r") as f:
                    trace_data = json.load(f)

            optimized_op_type_map, optimized_dump = build_optimized_op_type_map(
                logger=self.logger,
                warn_if_missing=False,
            )
            if optimized_op_type_map:
                self.logger.info(
                    "Operator type mapping overlaid from optimized graph: %s",
                    optimized_dump,
                )
            else:
                self.logger.info(
                    "No optimized after_fusion dump found; falling back to runtime event names."
                )

            # 提取事件信息
            events = trace_data.get("traceEvents", [])

            # 统计每个算子的执行时间
            op_times = collections.defaultdict(float)
            op_counts = collections.defaultdict(int)

            for event in events:
                if event.get("ph") == "X":  # Complete events
                    op_name = event.get("name", "unknown")
                    if op_name in self._FILTERED_PROFILE_NAMES or any(
                        op_name.startswith(prefix)
                        for prefix in self._FILTERED_PROFILE_PREFIXES
                    ):
                        continue
                    duration = event.get("dur", 0)  # Duration in microseconds

                    # Convert to milliseconds for readability
                    op_times[op_name] += duration / 1000.0
                    op_counts[op_name] += 1

            # Store the timing data
            for op_name, total_time in op_times.items():
                inferred_op_type = optimized_op_type_map.get(op_name)
                if not inferred_op_type:
                    inferred_op_type = (
                        "Send" if op_name.startswith("_Send input ") else op_name
                    )
                self.operator_timings[op_name].append(
                    {
                        "op_type": inferred_op_type,
                        "total_time_ms": total_time,
                        "count": op_counts[op_name],
                        "avg_time_ms": total_time / op_counts[op_name],
                    }
                )

            self.logger.info(f"Parsed operator timings from {trace_file}")
            self.logger.info(f"Found {len(op_times)} unique operators")

        except Exception as e:
            self.logger.error(f"Error parsing trace file {trace_file}: {e}")

    def print_operator_timings(self):
        """打印算子时间统计"""
        if not self.operator_timings:
            self.logger.info("No operator timing data available")
            return

        self.logger.info("=" * 80)
        self.logger.info("OPERATOR EXECUTION TIME STATISTICS")
        self.logger.info("=" * 80)

        # Aggregate statistics
        all_op_stats = []
        for op_name, timing_data in self.operator_timings.items():
            # Calculate aggregated stats for this operator
            total_time = sum([item["total_time_ms"] for item in timing_data])
            total_count = sum([item["count"] for item in timing_data])
            avg_time = total_time / total_count if total_count > 0 else 0
            op_type = (
                timing_data[0].get("op_type", "unknown") if timing_data else "unknown"
            )

            all_op_stats.append(
                {
                    "name": op_name,
                    "op_type": op_type,
                    "total_time_ms": total_time,
                    "count": total_count,
                    "avg_time_ms": avg_time,
                }
            )

        # Sort by total time (descending)
        all_op_stats.sort(key=lambda x: x["total_time_ms"], reverse=True)

        # Print top operators by total time
        self.logger.info(
            f"{'Operator Name':<40} {'Op Type':<20} {'Total Time (ms)':<15} {'Count':<10} {'Avg Time (ms)':<15}"
        )
        self.logger.info("-" * 110)

        for stat in all_op_stats[:20]:  # Show top 20 operators
            self.logger.info(
                f"{stat['name']:<40} {stat['op_type']:<20} "
                f"{stat['total_time_ms']:<15.3f} {stat['count']:<10} {stat['avg_time_ms']:<15.3f}"
            )

        if len(all_op_stats) > 20:
            self.logger.info(f"... and {len(all_op_stats) - 20} more operators")

        # Save operator timing results
        op_timing_file = os.path.join(self.trace_dir, "operator_timings.json")
        with open(op_timing_file, "w") as f:
            json.dump(
                {
                    "operators": [
                        {
                            "name": stat["name"],
                            "op_type": stat["op_type"],
                            "total_time_ms": stat["total_time_ms"],
                            "count": stat["count"],
                            "avg_time_ms": stat["avg_time_ms"],
                        }
                        for stat in all_op_stats
                    ],
                    "summary": {
                        "total_operators": len(all_op_stats),
                        "top_10_total_time": sum(
                            stat["total_time_ms"] for stat in all_op_stats[:10]
                        ),
                        "all_operators_total_time": sum(
                            stat["total_time_ms"] for stat in all_op_stats
                        ),
                    },
                },
                f,
                indent=2,
            )

        self.logger.info(f"Operator timing results saved to: {op_timing_file}")
        self.logger.info("=" * 80)

    @tf.function
    def inference_step(self, inputs):
        """单步推理函数"""
        return self.model(inputs, training=False)

    def run_inference_with_profiling(self):
        """运行带性能分析的推理"""
        self.logger.info("Starting inference with profiling...")

        # 预热运行
        self.logger.info("Running warmup inference...")
        for _ in range(3):
            _ = self.inference_step((self.test_sparse, self.test_dense))

        # 确保设备同步
        if hasattr(tf, "musa") and tf.config.list_physical_devices("/device:MUSA:0"):
            tf.musa.synchronize()

        # 启动性能分析
        self.logger.info("Starting performance profiling...")
        try:
            tf.profiler.experimental.start(self.trace_dir)

            start_time = time.time()
            result = self.inference_step((self.test_sparse, self.test_dense))
            end_time = time.time()

            # 停止性能分析
            tf.profiler.experimental.stop()

            # 确保设备同步以获得准确的时间
            if hasattr(tf, "musa") and tf.config.list_physical_devices(
                "/device:MUSA:0"
            ):
                tf.musa.synchronize()

        except AttributeError:
            # 如果profiler不可用，回退到基本计时
            self.logger.warning("TensorFlow profiler not available, using basic timing")
            start_time = time.time()
            result = self.inference_step((self.test_sparse, self.test_dense))
            end_time = time.time()

            # 确保设备同步
            if hasattr(tf, "musa") and tf.config.list_physical_devices(
                "/device:MUSA:0"
            ):
                tf.musa.synchronize()

        inference_time = end_time - start_time
        self.logger.info(f"Inference completed in {inference_time:.4f} seconds")
        self.logger.info(
            f"Throughput: {self.batch_size / inference_time:.2f} samples/second"
        )
        self.logger.info(f"Result shape: {result.shape}, dtype: {result.dtype}")

        # 检查trace文件是否生成
        trace_files = []
        for root, dirs, files in os.walk(self.trace_dir):
            for file in files:
                if file.endswith(".json") or "trace" in file.lower():
                    trace_files.append(os.path.join(root, file))

        if trace_files:
            self.logger.info(f"Trace files generated:")
            for trace_file in trace_files:
                self.logger.info(f"  {trace_file}")
        else:
            self.logger.warning("No trace files found in trace directory")

        return result, inference_time

    def profile_whole_network_performance(self, warmup_rounds=5, profiling_rounds=20):
        """整网推理性能分析，包含预热和多次循环统计"""
        self.logger.info(f"Starting whole network performance profiling...")
        self.logger.info(
            f"Warmup rounds: {warmup_rounds}, Profiling rounds: {profiling_rounds}"
        )

        # 预热阶段
        self.logger.info("Starting warmup rounds...")
        for i in range(warmup_rounds):
            _ = self.inference_step((self.test_sparse, self.test_dense))
            if hasattr(tf, "musa") and tf.config.list_physical_devices(
                "/device:MUSA:0"
            ):
                tf.musa.synchronize()
            if (i + 1) % 5 == 0:
                self.logger.info(f"Warmup round {i + 1}/{warmup_rounds} completed")

        # 性能分析阶段
        self.logger.info("Starting profiling rounds...")
        times = []

        for i in range(profiling_rounds):
            # 确保设备同步
            if hasattr(tf, "musa") and tf.config.list_physical_devices(
                "/device:MUSA:0"
            ):
                tf.musa.synchronize()

            start_time = time.time()
            result = self.inference_step((self.test_sparse, self.test_dense))

            # 确保设备同步以获得准确的时间
            if hasattr(tf, "musa") and tf.config.list_physical_devices(
                "/device:MUSA:0"
            ):
                tf.musa.synchronize()

            end_time = time.time()
            iteration_time = end_time - start_time
            times.append(iteration_time)

            if (i + 1) % 5 == 0:
                self.logger.info(
                    f"Profiling round {i + 1}/{profiling_rounds} completed, "
                    f"time: {iteration_time:.4f}s"
                )

        # 计算统计信息
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        throughput_avg = self.batch_size / avg_time
        throughput_min = self.batch_size / max_time  # 最慢时间对应最低吞吐量
        throughput_max = self.batch_size / min_time  # 最快时间对应最高吞吐量

        self.logger.info("=" * 60)
        self.logger.info("PERFORMANCE PROFILING RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"Average inference time: {avg_time:.6f} seconds")
        self.logger.info(f"Min inference time: {min_time:.6f} seconds")
        self.logger.info(f"Max inference time: {max_time:.6f} seconds")
        self.logger.info(f"Average throughput: {throughput_avg:.2f} samples/second")
        self.logger.info(f"Max throughput: {throughput_max:.2f} samples/second")
        self.logger.info(f"Min throughput: {throughput_min:.2f} samples/second")
        self.logger.info(f"Standard deviation: {np.std(times):.6f} seconds")
        self.logger.info("=" * 60)

        # 保存性能分析结果
        perf_result = {
            "warmup_rounds": warmup_rounds,
            "profiling_rounds": profiling_rounds,
            "average_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "average_throughput": throughput_avg,
            "max_throughput": throughput_max,
            "min_throughput": throughput_min,
            "std_deviation": float(np.std(times)),
            "all_times": [float(t) for t in times],
            "batch_size": self.batch_size,
        }

        perf_file = self.log_mgr.save_json("performance_result.json", perf_result)
        self.logger.info(f"Performance results saved to: {perf_file}")

        return perf_result

    def analyze_model_ops(self):
        """分析模型中的算子"""
        self.logger.info("Analyzing model operations...")

        # 获取模型的所有层
        layers_info = []
        for i, layer in enumerate(self.model.layers):
            layer_info = {
                "index": i,
                "name": layer.name,
                "type": type(layer).__name__,
                "trainable": layer.trainable,
            }
            layers_info.append(layer_info)
            self.logger.info(f"Layer {i}: {layer.name} ({type(layer).__name__})")

        # 获取所有变量的设备信息
        var_devices = {}
        for var in self.model.trainable_variables:
            var_devices[var.name] = var.device

        self.logger.info(
            f"Total trainable variables: {len(self.model.trainable_variables)}"
        )
        device_counts = {}
        for device in var_devices.values():
            device_counts[device] = device_counts.get(device, 0) + 1

        self.logger.info("Variable device distribution:")
        for device, count in device_counts.items():
            self.logger.info(f"  {device}: {count} variables")

        return layers_info, var_devices

    def list_operator_devices(self):
        """列出模型中所有算子及其运行设备"""
        self.logger.info("Listing all operators and their running devices...")

        # 创建一个函数图来获取操作符信息
        @tf.function
        def get_concrete_function():
            return self.model((self.test_sparse, self.test_dense), training=False)

        # 获取具体的函数
        concrete_func = get_concrete_function.get_concrete_function()

        # 获取函数定义
        func_def = concrete_func.graph.as_graph_def()

        # 统计不同设备上的算子
        cpu_ops = []
        gpu_ops = []
        musa_ops = []
        other_ops = []

        for node in func_def.node:
            # print(f"[debug for timo] Operator Name: {node.name}, Op: {node.op}, Device: {node.device}")  # 调试输出

            op_device = node.device if node.device else "unspecified"
            op_info = {"name": node.name, "op": node.op, "device": op_device}

            if "CPU" in op_device.upper():
                cpu_ops.append(op_info)
            elif "MUSA" in op_device.upper() or "GPU" in op_device.upper():
                if "MUSA" in op_device.upper():
                    musa_ops.append(op_info)
                else:
                    gpu_ops.append(op_info)
            else:
                other_ops.append(op_info)

        # 打印统计信息
        self.logger.info(f"Operators running on CPU: {len(cpu_ops)}")
        for op in cpu_ops[:10]:  # 只显示前10个作为示例
            self.logger.info(f"  - {op['name']} ({op['op']})")
        if len(cpu_ops) > 10:
            self.logger.info(f"  ... and {len(cpu_ops) - 10} more CPU operators")

        self.logger.info(f"Operators running on MUSA: {len(musa_ops)}")
        for op in musa_ops[:10]:
            self.logger.info(f"  - {op['name']} ({op['op']})")
        if len(musa_ops) > 10:
            self.logger.info(f"  ... and {len(musa_ops) - 10} more MUSA operators")

        self.logger.info(f"Operators running on GPU: {len(gpu_ops)}")
        for op in gpu_ops[:10]:
            self.logger.info(f"  - {op['name']} ({op['op']})")
        if len(gpu_ops) > 10:
            self.logger.info(f"  ... and {len(gpu_ops) - 10} more GPU operators")

        self.logger.info(f"Operators on other devices: {len(other_ops)}")
        for op in other_ops[:10]:
            self.logger.info(f"  - {op['name']} ({op['op']}) - Device: {op['device']}")
        if len(other_ops) > 10:
            self.logger.info(
                f"  ... and {len(other_ops) - 10} more other device operators"
            )

        # 返回所有算子信息
        all_ops = {"cpu": cpu_ops, "musa": musa_ops, "gpu": gpu_ops, "other": other_ops}

        return all_ops

    def run_inference_only(self, warmup_rounds=5, inference_rounds=20):
        """仅运行warmup和循环inference，不进行其他分析"""
        self.logger.info(f"Starting inference-only mode...")
        self.logger.info(
            f"Warmup rounds: {warmup_rounds}, Inference rounds: {inference_rounds}"
        )

        # 预热阶段
        self.logger.info("Starting warmup rounds...")
        for i in range(warmup_rounds):
            _ = self.inference_step((self.test_sparse, self.test_dense))
            if hasattr(tf, "musa") and tf.config.list_physical_devices(
                "/device:MUSA:0"
            ):
                tf.musa.synchronize()
            if (i + 1) % 5 == 0:
                self.logger.info(f"Warmup round {i + 1}/{warmup_rounds} completed")

        # 推理阶段
        self.logger.info("Starting inference rounds...")
        times = []

        for i in range(inference_rounds):
            # 确保设备同步
            if hasattr(tf, "musa") and tf.config.list_physical_devices(
                "/device:MUSA:0"
            ):
                tf.musa.synchronize()

            start_time = time.time()
            result = self.inference_step((self.test_sparse, self.test_dense))

            # 确保设备同步以获得准确的时间
            if hasattr(tf, "musa") and tf.config.list_physical_devices(
                "/device:MUSA:0"
            ):
                tf.musa.synchronize()

            end_time = time.time()
            iteration_time = end_time - start_time
            times.append(iteration_time)

            if (i + 1) % 5 == 0:
                self.logger.info(
                    f"Inference round {i + 1}/{inference_rounds} completed, "
                    f"time: {iteration_time:.4f}s"
                )

        # 计算统计信息
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        throughput_avg = self.batch_size / avg_time
        throughput_min = self.batch_size / max_time  # 最慢时间对应最低吞吐量
        throughput_max = self.batch_size / min_time  # 最快时间对应最高吞吐量

        self.logger.info("=" * 60)
        self.logger.info("INFERENCE-ONLY RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"Average inference time: {avg_time:.6f} seconds")
        self.logger.info(f"Min inference time: {min_time:.6f} seconds")
        self.logger.info(f"Max inference time: {max_time:.6f} seconds")
        self.logger.info(f"Average throughput: {throughput_avg:.2f} samples/second")
        self.logger.info(f"Max throughput: {throughput_max:.2f} samples/second")
        self.logger.info(f"Min throughput: {throughput_min:.2f} samples/second")
        self.logger.info(f"Standard deviation: {np.std(times):.6f} seconds")
        self.logger.info("=" * 60)

        # 保存性能分析结果
        perf_result = {
            "warmup_rounds": warmup_rounds,
            "inference_rounds": inference_rounds,
            "average_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "average_throughput": throughput_avg,
            "max_throughput": throughput_max,
            "min_throughput": throughput_min,
            "std_deviation": float(np.std(times)),
            "all_times": [float(t) for t in times],
            "batch_size": self.batch_size,
        }

        perf_file = os.path.join(self.trace_dir, "inference_only_result.json")
        with open(perf_file, "w") as f:
            json.dump(perf_result, f, indent=2)

        self.logger.info(f"Inference-only results saved to: {perf_file}")

        return perf_result

    def run_comprehensive_analysis(self):
        """运行完整的分析"""
        self.logger.info("=" * 60)
        self.logger.info("WUKONG MODEL INFERENCE ANALYSIS")
        self.logger.info("=" * 60)

        # 1. 分析模型结构
        layers_info, var_devices = self.analyze_model_ops()

        # 2. 列出所有算子及其运行设备
        self.logger.info("Analyzing operator devices...")
        operator_devices = self.list_operator_devices()

        # 3. 获取设备信息
        device_type = self.get_op_device_info()
        # print(f"[debug for timo] Device used for inference: {device_type}")

        # 4. 运行推理并性能分析
        result, inference_time = self.run_inference_with_profiling()

        # 5. 运行整网推理性能分析
        perf_result = self.profile_whole_network_performance(
            warmup_rounds=5, profiling_rounds=20
        )

        # 6. 保存分析结果
        analysis_result = {
            "timestamp": datetime.now().isoformat(),
            "batch_size": self.batch_size,
            "inference_time": inference_time,
            "throughput": self.batch_size / inference_time,
            "result_shape": result.shape.as_list(),
            "result_dtype": str(result.dtype),
            "layers_count": len(layers_info),
            "trainable_variables_count": len(var_devices),
            "device_type": device_type,
            "available_devices": [d.name for d in tf.config.list_physical_devices()],
            "operator_device_summary": {
                "cpu_ops_count": len(operator_devices["cpu"]),
                "musa_ops_count": len(operator_devices["musa"]),
                "gpu_ops_count": len(operator_devices["gpu"]),
                "other_ops_count": len(operator_devices["other"]),
            },
            "performance_profile": perf_result,
        }

        result_file = os.path.join(self.trace_dir, "analysis_result.json")
        with open(result_file, "w") as f:
            json.dump(analysis_result, f, indent=2)

        self.logger.info(f"Analysis results saved to: {result_file}")
        self.logger.info("=" * 60)
        self.logger.info("ANALYSIS COMPLETED")
        self.logger.info("=" * 60)

        return analysis_result


class AccuracyComparator:
    """CPU vs MUSA 精度对比器（Wukong 模型）"""

    def __init__(self, profiler: InferenceProfiler):
        self.profiler = profiler
        self.logger = profiler.logger
        self.trace_dir = profiler.trace_dir

    def run_on_device(self, device_type: str, warmup_rounds: int = 2) -> np.ndarray:
        """在指定设备上运行推理并返回结果"""
        self.logger.info(f"  在 {device_type} 上运行推理...")

        inputs = (self.profiler.test_sparse, self.profiler.test_dense)

        device_str = f"/device:{device_type}:0" if device_type == "MUSA" else "/CPU:0"

        # 预热
        for _ in range(warmup_rounds):
            with tf.device(device_str):
                _ = self.profiler.model(inputs, training=False)

        # 正式推理
        with tf.device(device_str):
            result = self.profiler.model(inputs, training=False)

        result_np = result.numpy()
        self.logger.info(
            f"  {device_type} 推理完成, shape={result_np.shape}, dtype={result_np.dtype}"
        )
        return result_np

    def compare_results(
        self,
        cpu_result: np.ndarray,
        musa_result: np.ndarray,
        rtol: float = 1e-5,
        atol: float = 1e-6,
    ) -> Dict[str, Any]:
        """比较 CPU 和 MUSA 的推理结果"""
        report = {
            "cpu_shape": list(cpu_result.shape),
            "musa_shape": list(musa_result.shape),
            "cpu_dtype": str(cpu_result.dtype),
            "musa_dtype": str(musa_result.dtype),
            "rtol": rtol,
            "atol": atol,
        }

        if cpu_result.shape != musa_result.shape:
            report["shape_match"] = False
            report["passed"] = False
            report[
                "error"
            ] = f"Shape 不匹配: CPU={cpu_result.shape}, MUSA={musa_result.shape}"
            return report
        report["shape_match"] = True

        cpu_f64 = cpu_result.astype(np.float64)
        musa_f64 = musa_result.astype(np.float64)

        # 绝对误差
        abs_diff = np.abs(cpu_f64 - musa_f64)
        report["max_abs_diff"] = float(np.max(abs_diff))
        report["mean_abs_diff"] = float(np.mean(abs_diff))
        report["median_abs_diff"] = float(np.median(abs_diff))

        # 相对误差
        denom = np.maximum(np.abs(cpu_f64), 1e-12)
        rel_diff = abs_diff / denom
        report["max_rel_diff"] = float(np.max(rel_diff))
        report["mean_rel_diff"] = float(np.mean(rel_diff))
        report["median_rel_diff"] = float(np.median(rel_diff))

        # allclose
        report["allclose"] = bool(np.allclose(cpu_f64, musa_f64, rtol=rtol, atol=atol))

        # 不匹配元素
        mismatch_mask = ~np.isclose(cpu_f64, musa_f64, rtol=rtol, atol=atol)
        num_mismatch = int(np.sum(mismatch_mask))
        total_elements = int(cpu_f64.size)
        report["num_mismatch"] = num_mismatch
        report["total_elements"] = total_elements
        report["mismatch_ratio"] = (
            num_mismatch / total_elements if total_elements > 0 else 0.0
        )

        # 数值统计
        report["cpu_stats"] = {
            "min": float(np.min(cpu_f64)),
            "max": float(np.max(cpu_f64)),
            "mean": float(np.mean(cpu_f64)),
            "std": float(np.std(cpu_f64)),
        }
        report["musa_stats"] = {
            "min": float(np.min(musa_f64)),
            "max": float(np.max(musa_f64)),
            "mean": float(np.mean(musa_f64)),
            "std": float(np.std(musa_f64)),
        }

        # 余弦相似度
        cpu_flat = cpu_f64.flatten()
        musa_flat = musa_f64.flatten()
        norm_cpu = np.linalg.norm(cpu_flat)
        norm_musa = np.linalg.norm(musa_flat)
        if norm_cpu > 0 and norm_musa > 0:
            report["cosine_similarity"] = float(
                np.dot(cpu_flat, musa_flat) / (norm_cpu * norm_musa)
            )
        else:
            report["cosine_similarity"] = (
                1.0 if np.allclose(cpu_flat, musa_flat) else 0.0
            )

        # 分档通过率
        thresholds = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        report["threshold_pass_rates"] = {}
        for t in thresholds:
            pass_count = int(np.sum(np.isclose(cpu_f64, musa_f64, rtol=t, atol=t)))
            report["threshold_pass_rates"][str(t)] = (
                pass_count / total_elements if total_elements > 0 else 1.0
            )

        report["passed"] = report["allclose"]
        return report

    def print_report(self, report: Dict[str, Any]) -> None:
        """打印精度对比报告"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("  CPU vs MUSA 精度对比报告")
        self.logger.info("=" * 80)

        passed_str = "✅ PASSED" if report.get("passed") else "❌ FAILED"
        self.logger.info(f"  结果: {passed_str}")
        self.logger.info(
            f"  CPU  Shape: {report['cpu_shape']}, Dtype: {report['cpu_dtype']}"
        )
        self.logger.info(
            f"  MUSA Shape: {report['musa_shape']}, Dtype: {report['musa_dtype']}"
        )
        self.logger.info(f"  容差: rtol={report['rtol']}, atol={report['atol']}")

        if not report.get("shape_match"):
            self.logger.error(f"  错误: {report.get('error')}")
            return

        self.logger.info("")
        self.logger.info(f"  {'指标':<20} {'值':>15}")
        self.logger.info(f"  {'-'*36}")
        self.logger.info(f"  {'最大绝对误差':<20} {report['max_abs_diff']:>15.2e}")
        self.logger.info(f"  {'平均绝对误差':<20} {report['mean_abs_diff']:>15.2e}")
        self.logger.info(f"  {'中位绝对误差':<20} {report['median_abs_diff']:>15.2e}")
        self.logger.info(f"  {'最大相对误差':<20} {report['max_rel_diff']:>15.2e}")
        self.logger.info(f"  {'平均相对误差':<20} {report['mean_rel_diff']:>15.2e}")
        self.logger.info(f"  {'余弦相似度':<20} {report['cosine_similarity']:>15.10f}")
        self.logger.info(f"  {'np.allclose':<20} {str(report['allclose']):>15}")
        self.logger.info(
            f"  {'不匹配元素':<20} {report['num_mismatch']:>10} / {report['total_elements']}"
        )
        self.logger.info(f"  {'不匹配比例':<20} {report['mismatch_ratio']:>15.6%}")

        # self.logger.info("")
        # self.logger.info("  各容差阈值下的通过率:")
        # for t_str, rate in report['threshold_pass_rates'].items():
        #     self.logger.info(f"    阈值 {t_str:>8}: {rate:.6%}")

        self.logger.info("")
        self.logger.info("  数值分布对比:")
        self.logger.info(f"  {'统计量':<8} {'CPU':>14} {'MUSA':>14} {'差异':>14}")
        self.logger.info(f"  {'-'*52}")
        for key in ["min", "max", "mean", "std"]:
            cpu_val = report["cpu_stats"][key]
            musa_val = report["musa_stats"][key]
            diff = abs(cpu_val - musa_val)
            self.logger.info(
                f"  {key:<8} {cpu_val:>14.6f} {musa_val:>14.6f} {diff:>14.2e}"
            )

        self.logger.info("=" * 80)

    def run_comparison(
        self, rtol: float = 1e-5, atol: float = 1e-6, warmup_rounds: int = 2
    ) -> Dict[str, Any]:
        """执行完整的 CPU vs MUSA 精度对比"""
        self.logger.info("=" * 80)
        self.logger.info("  CPU vs MUSA 精度对比验证 (Wukong Model)")
        self.logger.info("=" * 80)

        try:
            cpu_result = self.run_on_device("CPU", warmup_rounds)
        except Exception as e:
            self.logger.error(f"CPU 推理失败: {e}")
            import traceback

            traceback.print_exc()
            return {"passed": False, "error": f"CPU inference failed: {e}"}

        try:
            musa_result = self.run_on_device("MUSA", warmup_rounds)
        except Exception as e:
            self.logger.error(f"MUSA 推理失败: {e}")
            import traceback

            traceback.print_exc()
            return {"passed": False, "error": f"MUSA inference failed: {e}"}

        report = self.compare_results(cpu_result, musa_result, rtol=rtol, atol=atol)
        self.print_report(report)

        # 保存报告
        timestamp = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
        filepath = os.path.join(self.trace_dir, f"accuracy_comparison_{timestamp}.json")
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        self.logger.info(f"  精度对比报告已保存到: {filepath}")

        return report


def main():
    """主函数"""
    if not TF_AVAILABLE:
        print("TensorFlow is not available. Exiting.")
        return

    # 设置根日志（在创建 InferenceProfiler 之前）
    log_mgr = get_log_manager("wukong_inference")
    logger = log_mgr.get_logger("main")

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Wukong Model TensorFlow Inference Script - Support CPU/MUSA device comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 在 MUSA 设备上运行
  python run_wukong_tf_musa.py --device musa

  # 在 CPU 上运行（用于性能对比）
  python run_wukong_tf_musa.py --device cpu

  # 自定义配置
  python run_wukong_tf_musa.py --device musa --batch-size 512 --warmup-rounds 10 --inference-rounds 50

  # 手动指定相邻工作区中的插件路径（推荐）
  python run_wukong_tf_musa.py --device musa \
    --musa-plugin ../tensorflow_musa_extension/build/libmusa_plugin.so

  # 手动指定 Docker 中的绝对路径
  python run_wukong_tf_musa.py --device musa \
    --musa-plugin /workspace/tensorflow_musa_extension/build/libmusa_plugin.so
        """,
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Run only warmup and inference cycles without other analysis",
    )
    parser.add_argument(
        "--profile-ops",
        action="store_true",
        help="Profile individual operator execution times",
    )
    parser.add_argument(
        "--compare-accuracy", action="store_true", help="运行 CPU vs MUSA 精度对比验证"
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-2, help="精度对比的相对容差 (default: 1e-2)"
    )
    parser.add_argument(
        "--atol", type=float, default=1e-2, help="精度对比的绝对容差 (default: 1e-2)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for inference (default: 1024)",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=5,
        help="Number of warmup rounds (default: 5)",
    )
    parser.add_argument(
        "--inference-rounds",
        type=int,
        default=20,
        help="Number of inference rounds (default: 20)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "musa"],
        default="musa",
        help="Device to run inference: cpu or musa (default: musa)",
    )
    parser.add_argument(
        "--musa-plugin",
        type=str,
        default=get_default_musa_plugin_path(),
        help="Path to MUSA plugin library. You can pass either a relative "
        "sibling-workspace path or an absolute docker path; when omitted, "
        "the script auto-detects one.",
    )
    parser.add_argument(
        "--log-device-placement",
        action="store_true",
        help="Log device placement for each operation (default: False)",
    )

    args = parser.parse_args()
    args.musa_plugin = resolve_musa_plugin_path(args.musa_plugin)
    logger.info(f"Resolved MUSA plugin path: {args.musa_plugin}")

    # 加载 MUSA 插件（仅在 device=musa 时需要）
    if args.device == "musa" or args.compare_accuracy:
        if os.path.exists(args.musa_plugin):
            try:
                tf.load_library(args.musa_plugin)
                logger.info(
                    f">>>> [MUSA] Plugin loaded successfully from: {args.musa_plugin}"
                )
            except Exception as e:
                logger.error(f"!!!! [MUSA] Failed to load plugin: {e}")
                if args.compare_accuracy:
                    logger.error("精度对比模式需要 MUSA 插件，退出")
                    return
        else:
            logger.error(f"!!!! [MUSA] Plugin not found at {args.musa_plugin}")
            if args.compare_accuracy:
                logger.error("精度对比模式需要 MUSA 插件，退出")
                return
    else:
        logger.info("Running on CPU, MUSA plugin not loaded")

    # 设置设备日志（默认关闭，避免大量打印）
    tf.debugging.set_log_device_placement(args.log_device_placement)

    # 创建推理分析器
    profiler = InferenceProfiler(batch_size=args.batch_size, device_type=args.device)
    profiler.profile_ops = args.profile_ops

    # ==========================================
    # 精度对比模式
    # ==========================================
    if args.compare_accuracy:
        logger.info("\n" + "=" * 80)
        logger.info("  运行模式: CPU vs MUSA 精度对比验证 (Wukong)")
        logger.info("=" * 80)

        comparator = AccuracyComparator(profiler)
        report = comparator.run_comparison(
            rtol=args.rtol, atol=args.atol, warmup_rounds=args.warmup_rounds
        )

        if report.get("passed"):
            logger.info("\n🎉 精度对比通过！CPU 与 MUSA 结果一致。")
        else:
            logger.warning("\n⚠️  精度对比未通过，请检查报告中的详细信息。")

        logger.info(f"结果保存在: {profiler.trace_dir}")

        # 如果未指定其他模式则直接返回
        if not args.inference_only and not args.profile_ops:
            return

    if args.inference_only:
        # 仅运行 warmup 和 inference
        if args.profile_ops:
            # Run operator profiling
            result, inference_time = profiler.profile_operator_times(
                (profiler.test_sparse, profiler.test_dense)
            )

            logger.info(f"\nOperator profiling completed successfully!")
            logger.info(f"Total inference time: {inference_time:.6f} seconds")
            logger.info(
                f"Throughput: {profiler.batch_size / inference_time:.2f} samples/second"
            )
            logger.info(f"Results saved in: {profiler.trace_dir}")

            # Print operator timing results
            profiler.print_operator_timings()
        else:
            result = profiler.run_inference_only(
                warmup_rounds=args.warmup_rounds, inference_rounds=args.inference_rounds
            )

            logger.info(f"\nInference-only mode completed successfully!")
            logger.info(f"Average inference time: {result['average_time']:.6f} seconds")
            logger.info(
                f"Average throughput: {result['average_throughput']:.2f} samples/second"
            )
            logger.info(
                f"Min/Max throughput: {result['min_throughput']:.2f}/{result['max_throughput']:.2f} samples/second"
            )
            logger.info(f"Results saved in: {profiler.trace_dir}")
    else:
        if args.profile_ops:
            # Run comprehensive analysis with operator profiling
            logger.info("Running comprehensive analysis with operator profiling...")

            # Run operator profiling
            result, inference_time = profiler.profile_operator_times(
                (profiler.test_sparse, profiler.test_dense)
            )

            # Print operator timing results
            profiler.print_operator_timings()

            logger.info(f"\nOperator profiling completed successfully!")
            logger.info(f"Total inference time: {inference_time:.6f} seconds")
            logger.info(
                f"Throughput: {profiler.batch_size / inference_time:.2f} samples/second"
            )
            logger.info(f"Results saved in: {profiler.trace_dir}")
        else:
            # 运行完整分析
            result = profiler.run_comprehensive_analysis()

            logger.info(f"\nInference completed successfully!")
            logger.info(f"Total time: {result['inference_time']:.4f} seconds")
            logger.info(f"Throughput: {result['throughput']:.2f} samples/second")
            logger.info(f"Device used: {result['device_type']}")

            # 打印算子设备分布摘要
            op_summary = result["operator_device_summary"]
            logger.info(f"Operator distribution:")
            logger.info(f"  CPU ops: {op_summary['cpu_ops_count']}")
            logger.info(f"  MUSA ops: {op_summary['musa_ops_count']}")
            logger.info(f"  GPU ops: {op_summary['gpu_ops_count']}")
            logger.info(f"  Other ops: {op_summary['other_ops_count']}")

            # 打印性能分析摘要
            perf_summary = result["performance_profile"]
            logger.info(f"Performance profiling results:")
            logger.info(
                f"  Average inference time: {perf_summary['average_time']:.6f} seconds"
            )
            logger.info(
                f"  Average throughput: {perf_summary['average_throughput']:.2f} samples/second"
            )
            logger.info(
                f"  Min/Max throughput: {perf_summary['min_throughput']:.2f}/{perf_summary['max_throughput']:.2f} samples/second"
            )

            logger.info(f"Trace files are saved in: {profiler.trace_dir}")


if __name__ == "__main__":
    main()
