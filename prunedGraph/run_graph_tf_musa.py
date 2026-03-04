#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Graph Def 推理脚本
支持从 pb 文件加载图模型并在 CPU、MUSA 或 CUDA 设备上运行推理
支持 CPU/MUSA/CUDA 设备性能对比
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
import collections

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tensorflow.core.framework import graph_pb2
import numpy as np

try:
    from prettytable import PrettyTable
    PRETTYTABLE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: prettytable not available: {e}")
    PRETTYTABLE_AVAILABLE = False

# 禁用 V2 行为，确保 TF1 图能正常运行
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


# ==========================================
# 全局配置
# ==========================================
DEFAULT_MODEL_PATH = "./graph_def.pb"
DEFAULT_BATCH_SIZE = 100
DEFAULT_OUTPUT_NODE_NAME = "predicts"
DEFAULT_MUSA_PLUGIN_PATH = "../../tensorflow_musa_extension/build/libmusa_plugin.so"
DEFAULT_WARMUP_ROUNDS = 5
DEFAULT_INFERENCE_ROUNDS = 20


class GraphProfiler:
    """Graph 推理性能分析器"""

    def __init__(self, graph: tf.Graph, feed_dict: Dict, output_tensor: tf.Tensor,
                 batch_size: int = 100, device_type: str = "MUSA"):
        """初始化性能分析器

        Args:
            graph: TensorFlow 图对象
            feed_dict: 输入数据字典
            output_tensor: 输出张量
            batch_size: 批次大小
            device_type: 设备类型 (CPU/MUSA/CUDA)
        """
        self.graph = graph
        self.feed_dict = feed_dict
        self.output_tensor = output_tensor
        self.batch_size = batch_size
        self.device_type = device_type.upper()
        self.operator_timings = collections.defaultdict(list)
        self.setup_logging()

    def setup_logging(self):
        """设置日志记录"""
        log_dir = "logs/graph_inference"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")

        # 先创建 trace_dir，日志文件都放在 trace 文件夹下
        self.trace_dir = f"{log_dir}/{timestamp}_trace"
        os.makedirs(self.trace_dir, exist_ok=True)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 日志文件放在 trace 文件夹下
        file_handler = logging.FileHandler(
            f"{self.trace_dir}/inference.log", mode="a", encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(formatter)

        self.logger = logging.getLogger("graph_inference.profiler")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stdout_handler)

    def run_inference_only(self, warmup_rounds: int = 5, inference_rounds: int = 20) -> Dict[str, Any]:
        """仅运行 warmup 和 inference，不进行其他分析

        Args:
            warmup_rounds: 预热轮数
            inference_rounds: 推理轮数

        Returns:
            性能统计结果字典
        """
        self.logger.info("=" * 60)
        self.logger.info(f"INFERENCE-ONLY MODE (Device: {self.device_type})")
        self.logger.info("=" * 60)
        self.logger.info(f"Warmup rounds: {warmup_rounds}, Inference rounds: {inference_rounds}")

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.log_device_placement = False
        
        # 配置 GPU/CUDA 选项
        if self.device_type == "CUDA":
            config.gpu_options.allow_growth = True

        with tf.compat.v1.Session(graph=self.graph, config=config) as sess:
            # 预热阶段
            self.logger.info("\nStarting warmup rounds...")
            for i in range(warmup_rounds):
                sess.run(self.output_tensor, feed_dict=self.feed_dict)
                if (i + 1) % 5 == 0:
                    self.logger.info(f"  Warmup round {i + 1}/{warmup_rounds} completed")

            # 推理阶段
            self.logger.info("\nStarting inference rounds...")
            times = []

            for i in range(inference_rounds):
                start_time = time.time()
                sess.run(self.output_tensor, feed_dict=self.feed_dict)
                end_time = time.time()
                iteration_time = end_time - start_time
                times.append(iteration_time)

                if (i + 1) % 5 == 0:
                    self.logger.info(f"  Inference round {i + 1}/{inference_rounds} completed, "
                                    f"time: {iteration_time:.4f}s")

        # 计算统计信息
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        throughput_avg = self.batch_size / avg_time
        throughput_min = self.batch_size / max_time
        throughput_max = self.batch_size / min_time

        self.logger.info("\n" + "=" * 60)
        self.logger.info("INFERENCE-ONLY RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"Average inference time: {avg_time:.6f} seconds")
        self.logger.info(f"Min inference time:     {min_time:.6f} seconds")
        self.logger.info(f"Max inference time:     {max_time:.6f} seconds")
        self.logger.info(f"Average throughput:     {throughput_avg:.2f} samples/second")
        self.logger.info(f"Max throughput:         {throughput_max:.2f} samples/second")
        self.logger.info(f"Min throughput:         {throughput_min:.2f} samples/second")
        self.logger.info(f"Standard deviation:     {np.std(times):.6f} seconds")
        self.logger.info("=" * 60)

        # 保存结果
        perf_result = {
            'device_type': self.device_type,
            'warmup_rounds': warmup_rounds,
            'inference_rounds': inference_rounds,
            'average_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'average_throughput': throughput_avg,
            'max_throughput': throughput_max,
            'min_throughput': throughput_min,
            'std_deviation': float(np.std(times)),
            'all_times': [float(t) for t in times],
            'batch_size': self.batch_size
        }

        timestamp = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
        perf_file = os.path.join(self.trace_dir, f"inference_only_result_{self.device_type}_{timestamp}.json")
        with open(perf_file, 'w') as f:
            json.dump(perf_result, f, indent=2)

        self.logger.info(f"\nResults saved to: {perf_file}")

        return perf_result

    def profile_whole_network(self, warmup_rounds: int = 5, profiling_rounds: int = 20) -> Dict[str, Any]:
        """整网性能分析

        Args:
            warmup_rounds: 预热轮数
            profiling_rounds: 分析轮数

        Returns:
            性能分析结果字典
        """
        self.logger.info("=" * 60)
        self.logger.info(f"整网性能分析 (Device: {self.device_type})")
        self.logger.info("=" * 60)

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.log_device_placement = False
        
        # 配置 GPU/CUDA 选项
        if self.device_type == "CUDA":
            config.gpu_options.allow_growth = True

        with tf.compat.v1.Session(graph=self.graph, config=config) as sess:
            # 预热阶段
            self.logger.info(f"\n预热阶段：{warmup_rounds} 轮...")
            for i in range(warmup_rounds):
                sess.run(self.output_tensor, feed_dict=self.feed_dict)
                if (i + 1) % 5 == 0:
                    self.logger.info(f"  预热轮次 {i + 1}/{warmup_rounds} 完成")

            # 性能分析阶段
            self.logger.info(f"\n性能分析阶段：{profiling_rounds} 轮...")
            times = []

            for i in range(profiling_rounds):
                start_time = time.time()
                sess.run(self.output_tensor, feed_dict=self.feed_dict)
                end_time = time.time()
                iteration_time = end_time - start_time
                times.append(iteration_time)

                if (i + 1) % 5 == 0:
                    self.logger.info(f"  分析轮次 {i + 1}/{profiling_rounds} 完成，耗时：{iteration_time:.4f}s")

        # 计算统计信息
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        throughput_avg = self.batch_size / avg_time
        throughput_min = self.batch_size / max_time
        throughput_max = self.batch_size / min_time

        self.logger.info("\n" + "=" * 60)
        self.logger.info("整网性能分析结果")
        self.logger.info("=" * 60)
        self.logger.info(f"平均推理时间：    {avg_time:.6f} 秒")
        self.logger.info(f"最小推理时间：    {min_time:.6f} 秒")
        self.logger.info(f"最大推理时间：    {max_time:.6f} 秒")
        self.logger.info(f"平均吞吐量：      {throughput_avg:.2f} samples/秒")
        self.logger.info(f"最大吞吐量：      {throughput_max:.2f} samples/秒")
        self.logger.info(f"最小吞吐量：      {throughput_min:.2f} samples/秒")
        self.logger.info(f"标准差：          {np.std(times):.6f} 秒")
        self.logger.info("=" * 60)

        # 保存结果
        perf_result = {
            'device_type': self.device_type,
            'warmup_rounds': warmup_rounds,
            'profiling_rounds': profiling_rounds,
            'average_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'average_throughput': throughput_avg,
            'max_throughput': throughput_max,
            'min_throughput': throughput_min,
            'std_deviation': float(np.std(times)),
            'all_times': [float(t) for t in times],
            'batch_size': self.batch_size
        }

        timestamp = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
        perf_file = os.path.join(self.trace_dir, f"performance_result_{self.device_type}_{timestamp}.json")
        with open(perf_file, 'w') as f:
            json.dump(perf_result, f, indent=2)

        self.logger.info(f"\n性能结果已保存到：{perf_file}")

        return perf_result

    def profile_operator_times(self, warmup_rounds: int = 3) -> None:
        """单算子性能分析（使用 TensorFlow Profiler）

        Args:
            warmup_rounds: 预热轮数
        """
        self.logger.info("=" * 60)
        self.logger.info("单算子性能分析 (Operator Performance)")
        self.logger.info("=" * 60)

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.log_device_placement = False
        
        # 配置 GPU/CUDA 选项
        if self.device_type == "CUDA":
            config.gpu_options.allow_growth = True

        timestamp = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
        log_dir = os.path.join(self.trace_dir, f"ops_profile_{self.device_type}_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)

        with tf.compat.v1.Session(graph=self.graph, config=config) as sess:
            # 预热
            self.logger.info(f"\n预热：{warmup_rounds} 轮...")
            for _ in range(warmup_rounds):
                sess.run(self.output_tensor, feed_dict=self.feed_dict)

            # 启动 Profiler - 使用 TF1 兼容的 API
            self.logger.info("启动 TensorFlow Profiler...")
            try:
                # 使用 TF1 的 tf.profiler API
                run_meta = tf.RunMetadata()
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

                # 执行推理并收集 trace 信息
                self.logger.info("执行推理并收集性能数据...")
                result = sess.run(
                    self.output_tensor,
                    feed_dict=self.feed_dict,
                    options=run_options,
                    run_metadata=run_meta
                )

                # 导出 trace 数据
                from tensorflow.python.client import timeline
                timeline_obj = timeline.Timeline(run_meta.step_stats)
                trace_data = timeline_obj.generate_chrome_trace_format()

                # 保存 trace 数据
                trace_file = os.path.join(log_dir, f"trace_{self.device_type}_{timestamp}.json")
                with open(trace_file, 'w') as f:
                    f.write(trace_data)

                self.logger.info(f"Profiler 完成，trace 数据已保存到：{trace_file}")

                # 解析并打印算子统计信息
                self._print_op_stats_from_run_metadata(run_meta)

            except Exception as e:
                self.logger.warning(f"Profiler 不可用或出错：{e}")
                self.logger.info("尝试使用备用方法收集算子信息...")
                # 备用方法：直接执行并计时
                self._profile_with_simple_timing(sess, warmup_rounds)

    def _print_op_stats_from_run_metadata(self, run_meta: tf.RunMetadata) -> None:
        """从 RunMetadata 打印算子统计信息

        Args:
            run_meta: TensorFlow RunMetadata 对象
        """
        if not run_meta or not run_meta.step_stats:
            self.logger.info("无算子性能数据")
            return

        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"算子执行时间统计 (Device: {self.device_type})")
        self.logger.info("=" * 80)

        # 收集算子统计信息
        op_stats = []
        for dev_stat in run_meta.step_stats.dev_stats:
            device_name = dev_stat.device
            for node_stat in dev_stat.node_stats:
                op_name = node_stat.node_name
                # 获取执行时间（微秒转换为毫秒）
                if node_stat.all_end_rel_micros:
                    duration_ms = node_stat.all_end_rel_micros / 1000.0
                    op_stats.append({
                        'name': op_name,
                        'device': device_name,
                        'duration_ms': duration_ms
                    })

        # 按执行时间排序
        op_stats.sort(key=lambda x: x['duration_ms'], reverse=True)

        # 使用 prettytable 打印表格
        if PRETTYTABLE_AVAILABLE and op_stats:
            # 使用 logger 输出表格，避免与其他日志输出穿插
            self.logger.info("\n" + "=" * 100)
            self.logger.info(f"算子执行时间统计 (Device: {self.device_type}, Top 30 by Duration)")
            self.logger.info("=" * 100)

            table = PrettyTable()
            table.field_names = ["Rank", "Operator Name", "Device", "Duration (ms)"]
            table.align["Operator Name"] = "l"
            table.align["Device"] = "l"
            table.align["Duration (ms)"] = "r"

            for i, stat in enumerate(op_stats[:30], 1):
                table.add_row([
                    i,
                    stat['name'][:45] if len(stat['name']) > 45 else stat['name'],
                    stat['device'].split('/')[-1] if '/' in stat['device'] else stat['device'],
                    f"{stat['duration_ms']:.3f}"
                ])

            # 将表格作为整体通过 logger 输出，避免穿插
            for line in table.get_string().split('\n'):
                self.logger.info(line)

            if len(op_stats) > 30:
                self.logger.info(f"... 还有 {len(op_stats) - 30} 个算子")
        else:
            # 降级到普通打印
            self.logger.info(f"{'Operator Name':<50} {'Device':<20} {'Duration(ms)':<12}")
            self.logger.info("-" * 80)
            for stat in op_stats[:30]:
                device_short = stat['device'].split('/')[-1] if '/' in stat['device'] else stat['device']
                self.logger.info(f"{stat['name']:<50} {device_short:<20} {stat['duration_ms']:<12.3f}")

        self.logger.info("=" * 80)

        # 保存结果（标记为 RunMetadata 来源，避免与 print_operator_timings 的保存消息重复）
        if op_stats and not self.operator_timings:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
            op_stats_file = os.path.join(self.trace_dir, f"op_stats_{self.device_type}_{timestamp}.json")
            with open(op_stats_file, 'w') as f:
                json.dump({
                    'device_type': self.device_type,
                    'operators': op_stats,
                    'total_operators': len(op_stats)
                }, f, indent=2)
            self.logger.info(f"\n算子统计结果已保存到：{op_stats_file}")

    def parse_operator_timings(self, log_dir: str) -> None:
        """解析算子时间信息

        Args:
            log_dir: Profiler 输出目录
        """
        self.logger.info("\n解析算子性能数据...")

        trace_files = []
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.endswith('.json'):
                    trace_files.append(os.path.join(root, file))

        if not trace_files:
            self.logger.warning("未找到算子 trace 文件")
            return

        trace_file = trace_files[0]
        self.logger.info(f"解析文件：{trace_file}")

        try:
            with open(trace_file, 'r') as f:
                trace_data = json.load(f)

            events = trace_data.get('traceEvents', [])
            op_times = collections.defaultdict(float)
            op_counts = collections.defaultdict(int)

            for event in events:
                if event.get('ph') == 'X':
                    op_name = event.get('name', 'unknown')
                    duration = event.get('dur', 0) / 1000.0
                    op_times[op_name] += duration
                    op_counts[op_name] += 1

            for op_name, total_time in op_times.items():
                self.operator_timings[op_name].append({
                    'total_time_ms': total_time,
                    'count': op_counts[op_name],
                    'avg_time_ms': total_time / op_counts[op_name]
                })

            self.logger.info(f"解析完成，找到 {len(op_times)} 个算子")

        except Exception as e:
            self.logger.error(f"解析 trace 文件出错：{e}")
            return

        self.print_operator_timings()

    def print_operator_timings(self) -> None:
        """使用 prettytable 打印算子时间统计"""
        if not self.operator_timings:
            self.logger.info("无算子时间数据")
            return

        all_op_stats = []
        for op_name, timing_data in self.operator_timings.items():
            total_time = sum([item['total_time_ms'] for item in timing_data])
            total_count = sum([item['count'] for item in timing_data])
            avg_time = total_time / total_count if total_count > 0 else 0

            all_op_stats.append({
                'name': op_name,
                'total_time_ms': total_time,
                'count': total_count,
                'avg_time_ms': avg_time
            })

        all_op_stats.sort(key=lambda x: x['total_time_ms'], reverse=True)

        # 使用 prettytable 打印表格
        if PRETTYTABLE_AVAILABLE:
            # 使用 logger 输出表格，避免与其他日志输出穿插
            self.logger.info("\n" + "=" * 100)
            self.logger.info(f"算子执行时间统计 (Device: {self.device_type}, Top 30 by Total Time)")
            self.logger.info("=" * 100)

            table = PrettyTable()
            table.field_names = ["Rank", "Operator Name", "Total Time (ms)", "Count", "Avg Time (ms)"]
            table.align["Operator Name"] = "l"
            table.align["Total Time (ms)"] = "r"
            table.align["Count"] = "r"
            table.align["Avg Time (ms)"] = "r"

            for i, stat in enumerate(all_op_stats[:30], 1):
                table.add_row([
                    i,
                    stat['name'][:45] if len(stat['name']) > 45 else stat['name'],
                    f"{stat['total_time_ms']:.3f}",
                    stat['count'],
                    f"{stat['avg_time_ms']:.3f}"
                ])

            # 将表格作为整体通过 logger 输出，避免穿插
            for line in table.get_string().split('\n'):
                self.logger.info(line)

            if len(all_op_stats) > 30:
                self.logger.info(f"... 还有 {len(all_op_stats) - 30} 个算子")
        else:
            # 降级到普通打印
            self.logger.info("\n" + "=" * 80)
            self.logger.info(f"算子执行时间统计 (Device: {self.device_type}, Top 30 by Total Time)")
            self.logger.info("=" * 80)
            self.logger.info(f"{'Operator Name':<50} {'Total(ms)':<12} {'Count':<8} {'Avg(ms)':<12}")
            self.logger.info("-" * 80)
            for stat in all_op_stats[:30]:
                self.logger.info(f"{stat['name']:<50} {stat['total_time_ms']:<12.3f} {stat['count']:<8} {stat['avg_time_ms']:<12.3f}")

        # 保存结果
        timestamp = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
        op_timing_file = os.path.join(self.trace_dir, f"operator_timings_{self.device_type}_{timestamp}.json")
        with open(op_timing_file, 'w') as f:
            json.dump({
                'device_type': self.device_type,
                'operators': [
                    {
                        'name': stat['name'],
                        'total_time_ms': stat['total_time_ms'],
                        'count': stat['count'],
                        'avg_time_ms': stat['avg_time_ms']
                    } for stat in all_op_stats
                ],
                'summary': {
                    'total_operators': len(all_op_stats),
                    'top_10_total_time': sum(stat['total_time_ms'] for stat in all_op_stats[:10]),
                    'all_operators_total_time': sum(stat['total_time_ms'] for stat in all_op_stats)
                }
            }, f, indent=2)

        self.logger.info(f"\n算子性能结果已保存到：{op_timing_file}")
        self.logger.info("=" * 80)

    def print_performance_table(self, perf_result: Dict[str, Any]) -> None:
        """使用 prettytable 打印性能结果表格

        Args:
            perf_result: 性能分析结果字典
        """
        # Log to logger as well to ensure output is captured
        self.logger.info("=" * 60)
        self.logger.info("PERFORMANCE SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Device: {perf_result['device_type']}")
        self.logger.info(f"Batch Size: {perf_result['batch_size']}")
        self.logger.info(f"Average Time (s): {perf_result['average_time']:.6f}")
        self.logger.info(f"Min Time (s): {perf_result['min_time']:.6f}")
        self.logger.info(f"Max Time (s): {perf_result['max_time']:.6f}")
        self.logger.info(f"Avg Throughput (samples/s): {perf_result['average_throughput']:.2f}")
        self.logger.info(f"Max Throughput (samples/s): {perf_result['max_throughput']:.2f}")
        self.logger.info(f"Min Throughput (samples/s): {perf_result['min_throughput']:.2f}")
        self.logger.info(f"Std Deviation (s): {perf_result['std_deviation']:.6f}")
        self.logger.info("=" * 60)

        # Also print to stdout for terminal display
        if PRETTYTABLE_AVAILABLE:

            print("\n" + "=" * 60)
            print("PERFORMANCE SUMMARY")
            print("=" * 60)

            table = PrettyTable()
            table.field_names = ["Metric", "Value"]
            table.align["Metric"] = "l"
            table.align["Value"] = "r"

            table.add_row(["Device", perf_result['device_type']])
            table.add_row(["Batch Size", perf_result['batch_size']])
            table.add_row(["Average Time (s)", f"{perf_result['average_time']:.6f}"])
            table.add_row(["Min Time (s)", f"{perf_result['min_time']:.6f}"])
            table.add_row(["Max Time (s)", f"{perf_result['max_time']:.6f}"])
            table.add_row(["Avg Throughput (samples/s)", f"{perf_result['average_throughput']:.2f}"])
            table.add_row(["Max Throughput (samples/s)", f"{perf_result['max_throughput']:.2f}"])
            table.add_row(["Min Throughput (samples/s)", f"{perf_result['min_throughput']:.2f}"])
            table.add_row(["Std Deviation (s)", f"{perf_result['std_deviation']:.6f}"])

            print(table)
            print("=" * 60)
            sys.stdout.flush() # Ensure output is flushed to terminal

def infer_placeholder_shape_from_usage(graph_def: graph_pb2.GraphDef, placeholder_name: str) -> Optional[List[int]]:
    """通过分析图中使用该 Placeholder 的节点来推断其形状

    Args:
        graph_def: 图定义
        placeholder_name: Placeholder 名称

    Returns:
        推断的形状列表，如果无法推断则返回 None
    """
    for node in graph_def.node:
        for input_name in node.input:
            clean_input = input_name.split(":")[0].lstrip("^")
            if clean_input == placeholder_name:
                if node.op == "MatMul" or node.op == "Tensordot":
                    if "_output_shapes" in node.attr:
                        output_shapes = node.attr["_output_shapes"].list.shape
                        if len(output_shapes) > 0:
                            output_shape = output_shapes[0]
                            if len(output_shape.dim) == 2:
                                return [64, 32]
                elif node.op == "BiasAdd":
                    if "_output_shapes" in node.attr:
                        output_shapes = node.attr["_output_shapes"].list.shape
                        if len(output_shapes) > 0:
                            output_shape = output_shapes[0]
                            if len(output_shape.dim) >= 1:
                                return [output_shape.dim[-1].size]
    return None


def load_graph_and_get_placeholders(pb_path: str, logger: logging.Logger) -> tuple:
    """加载图并获取所有 placeholder 节点信息

    Args:
        pb_path: 图文件路径
        logger: 日志记录器

    Returns:
        (graph_def, placeholders) 元组
    """
    logger.info(f"\n=== 加载图文件：{pb_path} ===")

    if not os.path.exists(pb_path):
        logger.error(f"错误：文件 {pb_path} 不存在!")
        sys.exit(1)

    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def = graph_pb2.GraphDef()
        graph_def.ParseFromString(f.read())

    logger.info(f"图加载成功，总节点数：{len(graph_def.node)}")

    placeholders = {}
    for node in graph_def.node:
        if node.op == "Placeholder":
            dtype_enum = node.attr["dtype"].type
            dtype_map = {
                tf.float32.as_datatype_enum: np.float32,
                tf.int32.as_datatype_enum: np.int32,
                tf.int64.as_datatype_enum: np.int64,
                tf.bool.as_datatype_enum: np.bool_,
                tf.string.as_datatype_enum: np.str_,
            }
            dtype = dtype_map.get(dtype_enum, np.float32)

            shape = []
            shape_found = False

            if "shape" in node.attr:
                shape_proto = node.attr["shape"].shape
                if not shape_proto.unknown_rank:
                    for dim in shape_proto.dim:
                        shape.append(dim.size if dim.size != -1 else None)
                    shape_found = True

            if not shape_found and "_output_shapes" in node.attr:
                output_shapes = node.attr["_output_shapes"].list.shape
                if len(output_shapes) > 0:
                    shape_proto = output_shapes[0]
                    if not shape_proto.unknown_rank:
                        for dim in shape_proto.dim:
                            shape.append(dim.size if dim.size != -1 else None)
                        shape_found = True

            if not shape_found:
                inferred = infer_placeholder_shape_from_usage(graph_def, node.name)
                shape = inferred if inferred else []

            placeholders[node.name] = {"dtype": dtype, "shape": shape}

    logger.info(f"找到 {len(placeholders)} 个 Placeholder 节点")
    return graph_def, placeholders


def create_mock_data(placeholders: Dict[str, Dict], batch_size: int) -> Dict[str, np.ndarray]:
    """根据 placeholder 信息创建 mock 数据

    Args:
        placeholders: Placeholder 信息字典
        batch_size: 批次大小

    Returns:
        输入数据字典
    """
    logger = logging.getLogger("graph_inference.main")
    logger.info("\n=== 创建 Mock 数据 ===")

    feed_dict = {}

    for name, info in placeholders.items():
        shape = info["shape"]
        dtype = info["dtype"]

        mock_shape = []
        for dim in shape:
            if dim is None:
                mock_shape.append(batch_size)
            elif dim == 0:
                mock_shape.append(0)
            else:
                mock_shape.append(dim)

        if not mock_shape:
            if "/ReadVariableOp/resource" in name:
                if "BiasAdd" in name:
                    mock_shape = [32]
                elif "MatMul" in name or "Tensordot" in name:
                    mock_shape = [64, 32]
                else:
                    mock_shape = []
            else:
                mock_shape = []

        if dtype == np.float32:
            mock_data = np.random.normal(0.0, 1.0, mock_shape).astype(dtype)
        elif dtype == np.int32:
            mock_data = np.random.randint(0, 100, mock_shape).astype(dtype)
        elif dtype == np.int64:
            mock_data = np.random.randint(0, 100, mock_shape).astype(dtype)
        elif dtype == np.bool_:
            mock_data = np.random.choice([True, False], mock_shape).astype(dtype)
        else:
            mock_data = np.random.normal(0.0, 1.0, mock_shape).astype(np.float32)

        feed_dict[name + ":0"] = mock_data

    return feed_dict


def run_inference(graph_def: graph_pb2.GraphDef, feed_dict: Dict, output_node_name: str,
                  device_type: str, batch_size: int, enable_profiling: bool = True,
                  warmup_rounds: int = 5, inference_rounds: int = 20,
                  log_device_placement: bool = False) -> Optional[np.ndarray]:
    """执行图推理

    Args:
        graph_def: 图定义
        feed_dict: 输入数据字典
        output_node_name: 输出节点名称
        device_type: 设备类型
        batch_size: 批次大小
        enable_profiling: 是否启用性能分析
        warmup_rounds: 预热轮数
        inference_rounds: 推理轮数
        log_device_placement: 是否记录设备放置信息

    Returns:
        推理结果，如果失败则返回 None
    """
    logger = logging.getLogger("graph_inference.main")
    logger.info(f"\n=== 执行图推理 (Device: {device_type}) ===")
    logger.info(f"输出节点：{output_node_name}")

    devices = tf.config.list_physical_devices()
    logger.info(f"Available devices: {[d.name for d in devices]}")

    if device_type == "MUSA":
        musa_devices = tf.config.list_physical_devices('MUSA')
        logger.info(f"MUSA devices available: {len(musa_devices)}")
        if not musa_devices:
            logger.warning("!!!! 警告：未检测到 MUSA 设备，将回退到 CPU 运行")
    elif device_type == "CUDA":
        gpu_devices = tf.config.list_physical_devices('GPU')
        logger.info(f"CUDA GPU devices available: {len(gpu_devices)}")
        if not gpu_devices:
            logger.warning("!!!! 警告：未检测到 CUDA GPU 设备，将回退到 CPU 运行")

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

        session_feed_dict = {}
        for name, data in feed_dict.items():
            try:
                tensor = graph.get_tensor_by_name(name)
                session_feed_dict[tensor] = data
            except KeyError:
                pass

        try:
            output_tensor = graph.get_tensor_by_name(f"{output_node_name}:0")
        except KeyError:
            logger.error(f"错误：找不到输出张量 {output_node_name}:0")
            return None

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.log_device_placement = log_device_placement
        
        # 配置 GPU/CUDA 选项
        if device_type == "CUDA":
            config.gpu_options.allow_growth = True

        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            try:
                logger.info(">>> Session Run Start...")
                if device_type == "MUSA":
                    with tf.device("/device:MUSA:0"):
                        result = sess.run(output_tensor, feed_dict=session_feed_dict)
                elif device_type == "CUDA":
                    with tf.device("/GPU:0"):
                        result = sess.run(output_tensor, feed_dict=session_feed_dict)
                else:
                    result = sess.run(output_tensor, feed_dict=session_feed_dict)
                logger.info(">>> Session Run Success!")

                logger.info(f"\n[推理结果统计]")
                logger.info(f"  Shape: {result.shape}")
                logger.info(f"  Dtype: {result.dtype}")
                logger.info(f"  Min:   {np.min(result):.4f}")
                logger.info(f"  Max:   {np.max(result):.4f}")
                logger.info(f"  Mean:  {np.mean(result):.4f}")

                if result.size <= 20:
                    logger.info(f"  Data: {result}")

                return result

            except Exception as e:
                logger.error(f"\n!!!! 推理失败 !!!!")
                logger.error(f"错误信息：{e}")
                import traceback
                traceback.print_exc()
                return None


def main():
    """主函数"""
    # 设置根日志
    log_dir = "logs/graph_inference"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")

    # 先创建 trace_dir，日志文件都放在 trace 文件夹下
    trace_dir = f"{log_dir}/{timestamp}_trace"
    os.makedirs(trace_dir, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 日志文件放在 trace 文件夹下
    file_handler = logging.FileHandler(
        f"{trace_dir}/main.log", mode="a", encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    logger = logging.getLogger("graph_inference.main")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Graph Def TensorFlow 推理脚本 - 支持 CPU/MUSA/CUDA 设备性能对比',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 在 MUSA 设备上运行（完整分析）
  python run_graph_tf_musa.py --device musa

  # 在 CUDA GPU 上运行（完整分析）
  python run_graph_tf_musa.py --device cuda

  # 在 CPU 上运行（用于性能对比）
  python run_graph_tf_musa.py --device cpu

  # 仅运行 warmup 和 inference（不进行算子分析）
  python run_graph_tf_musa.py --device musa --inference-only

  # 仅分析算子性能
  python run_graph_tf_musa.py --device musa --profile-ops

  # 自定义配置
  python run_graph_tf_musa.py --device musa --batch-size 256 --warmup-rounds 10 --inference-rounds 50
        """
    )

    parser.add_argument('--inference-only', action='store_true',
                        help='仅运行 warmup 和 inference 轮次，不进行其他分析')
    parser.add_argument('--profile-ops', action='store_true',
                        help='分析单个算子的执行时间')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help=f'模型文件路径 (pb 格式)，默认：{DEFAULT_MODEL_PATH}')
    parser.add_argument('--output-node', type=str, default=DEFAULT_OUTPUT_NODE_NAME,
                        help=f'输出节点名称，默认：{DEFAULT_OUTPUT_NODE_NAME}')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'批次大小，默认：{DEFAULT_BATCH_SIZE}')
    parser.add_argument('--device', type=str, choices=['cpu', 'musa', 'cuda'], default='musa',
                        help='运行设备：cpu、musa 或 cuda，默认：musa')
    parser.add_argument('--musa-plugin', type=str, default=DEFAULT_MUSA_PLUGIN_PATH,
                        help=f'MUSA 插件路径，默认：{DEFAULT_MUSA_PLUGIN_PATH}')
    parser.add_argument('--log-device-placement', action='store_true',
                        help='记录每个算子的设备放置信息（默认：False）')
    parser.add_argument('--warmup-rounds', type=int, default=DEFAULT_WARMUP_ROUNDS,
                        help=f'预热轮数，默认：{DEFAULT_WARMUP_ROUNDS}')
    parser.add_argument('--inference-rounds', type=int, default=DEFAULT_INFERENCE_ROUNDS,
                        help=f'推理轮数，默认：{DEFAULT_INFERENCE_ROUNDS}')

    args = parser.parse_args()

    # 加载 MUSA 插件（仅在 device=musa 时需要）
    if args.device == 'musa':
        if os.path.exists(args.musa_plugin):
            try:
                tf.load_op_library(args.musa_plugin)
                logger.info(f">>>> [MUSA] Plugin loaded successfully from: {args.musa_plugin}")
            except Exception as e:
                logger.error(f"!!!! [MUSA] Failed to load plugin: {e}")
        else:
            logger.error(f"!!!! [MUSA] Plugin not found at {args.musa_plugin}")
    elif args.device == 'cuda':
        logger.info("Running on CUDA GPU, MUSA plugin not loaded")
    else:
        logger.info("Running on CPU, MUSA plugin not loaded")

    # 1. 分析图
    graph_def, placeholders = load_graph_and_get_placeholders(args.model, logger)
    if not placeholders:
        logger.error("错误：未找到 Placeholder")
        return

    # 2. 造数据
    feed_dict = create_mock_data(placeholders, args.batch_size)

    # 3. 创建图并准备推理
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

        session_feed_dict = {}
        for name, data in feed_dict.items():
            try:
                tensor = graph.get_tensor_by_name(name)
                session_feed_dict[tensor] = data
            except KeyError:
                pass

        try:
            output_tensor = graph.get_tensor_by_name(f"{args.output_node}:0")
        except KeyError:
            logger.error(f"错误：找不到输出张量 {args.output_node}:0")
            return

        # 创建性能分析器
        profiler = GraphProfiler(graph, session_feed_dict, output_tensor, args.batch_size, args.device.upper())

        # 根据参数执行不同的测试模式
        if args.inference_only:
            # 仅运行 warmup 和 inference
            if args.profile_ops:
                # 运行算子分析（仅使用 profile_operator_times，避免重复统计）
                logger.info("\nRunning inference with operator profiling...")
                run_inference(
                    graph_def=graph_def,
                    feed_dict=feed_dict,
                    output_node_name=args.output_node,
                    device_type=args.device.upper(),
                    batch_size=args.batch_size,
                    enable_profiling=False,
                    warmup_rounds=args.warmup_rounds,
                    inference_rounds=args.warmup_rounds,  # 预热
                    log_device_placement=args.log_device_placement
                )
                # profile_operator_times 内部会调用 _print_op_stats_from_run_metadata 打印和保存结果
                profiler.profile_operator_times(warmup_rounds=3)
            else:
                result = profiler.run_inference_only(
                    warmup_rounds=args.warmup_rounds,
                    inference_rounds=args.inference_rounds
                )
                profiler.print_performance_table(result)

            logger.info(f"\nInference-only mode completed!")
            logger.info(f"Results saved in: {profiler.trace_dir}")

        elif args.profile_ops:
            # 仅分析算子性能（profile_operator_times 内部已包含打印和保存逻辑）
            logger.info("\nRunning operator profiling...")
            profiler.profile_operator_times(warmup_rounds=args.warmup_rounds)

            logger.info(f"\nOperator profiling completed!")
            logger.info(f"Results saved in: {profiler.trace_dir}")

        else:
            # 运行完整分析
            logger.info("\nRunning comprehensive analysis...")

            # 先运行推理
            run_inference(
                graph_def=graph_def,
                feed_dict=feed_dict,
                output_node_name=args.output_node,
                device_type=args.device.upper(),
                batch_size=args.batch_size,
                enable_profiling=False,
                warmup_rounds=args.warmup_rounds,
                inference_rounds=args.warmup_rounds,
                log_device_placement=args.log_device_placement
            )

            # 整网性能分析
            perf_result = profiler.profile_whole_network(
                warmup_rounds=args.warmup_rounds,
                profiling_rounds=args.inference_rounds
            )
            profiler.print_performance_table(perf_result)

            # 算子性能分析（profile_operator_times 内部已包含打印和保存逻辑）
            profiler.profile_operator_times()

            logger.info(f"\nComprehensive analysis completed!")
            logger.info(f"Results saved in: {profiler.trace_dir}")


if __name__ == "__main__":
    main()
