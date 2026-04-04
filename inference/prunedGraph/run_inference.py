import argparse
import os
import sys
import time
import numpy as np
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inference_utils import (create_session_config,
                             check_acc,
                             load_musa_plugin,
                             setup_logger,
                             parse_arguments,
                             print_perf_result,
                             setup_environment,
                             set_random_seeds,
                             compare_accuracy,
                             create_parent_parser,
)

import tensorflow.compat.v1 as tf
from tensorflow.core.framework import graph_pb2

# 禁用 V2 行为，确保 TF1 图能正常运行
tf.disable_eager_execution()


# ==========================================
# 1. 创建自定义参数解析器
# ==========================================
def create_pruned_graph_parser():
    """创建prunedGraph特有的参数解析器"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--batchsize",
        type=int,
        default=100,
        help="batch size for inference (default: 100)"
    )
    parser.add_argument(
        "--check-acc",
        action="store_true",
        help="Compare results with CPU baseline for accuracy check",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for accuracy comparison (default: 1e-5)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for accuracy comparison (default: 1e-5)",
    )
    parser.add_argument(
        "--cossim",
        type=float,
        default=0.9999,
        help="Cosine similarity threshold for accuracy comparison (default: 0.9999)",
    )
    return parser


# ==========================================
# 2. 辅助函数：推断 Placeholder 形状
# ==========================================
def infer_placeholder_shape_from_usage(graph_def, placeholder_name):
    """
    通过分析图中使用该 Placeholder 的节点来推断其形状
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
                                # 简单推断逻辑
                                return [64, 32]  # 默认权重矩阵大小
                elif node.op == "BiasAdd":
                    if "_output_shapes" in node.attr:
                        output_shapes = node.attr["_output_shapes"].list.shape
                        if len(output_shapes) > 0:
                            output_shape = output_shapes[0]
                            if len(output_shape.dim) >= 1:
                                return [output_shape.dim[-1].size]
    return None


def load_graph_and_get_placeholders(pb_path, logger):
    """
    加载图并获取所有 placeholder 节点信息
    """
    logger.info(f"=== 加载图文件: {pb_path} ===")

    if not os.path.exists(pb_path):
        logger.info(f"错误: 文件 {pb_path} 不存在!")
        sys.exit(1)

    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def = graph_pb2.GraphDef()
        graph_def.ParseFromString(f.read())

    logger.info(f"图加载成功，总节点数: {len(graph_def.node)}")

    placeholders = {}
    for node in graph_def.node:
        if node.op == "Placeholder":
            # 获取数据类型
            dtype_enum = node.attr["dtype"].type
            dtype_map = {
                tf.float32.as_datatype_enum: np.float32,
                tf.int32.as_datatype_enum: np.int32,
                tf.int64.as_datatype_enum: np.int64,
                tf.bool.as_datatype_enum: np.bool_,
                tf.string.as_datatype_enum: np.str_,
            }
            dtype = dtype_map.get(dtype_enum, np.float32)

            # 获取形状
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

            # 兜底推断
            if not shape_found:
                inferred = infer_placeholder_shape_from_usage(graph_def, node.name)
                shape = inferred if inferred else []

            placeholders[node.name] = {"dtype": dtype, "shape": shape}

    # logger.info(f"找到 {len(placeholders)} 个 Placeholder 节点")
    return graph_def, placeholders


# ==========================================
# 3. 创建 Mock 数据 (含关键修复)
# ==========================================
def create_mock_data(placeholders, batch_size, logger):
    """
    根据 placeholder 信息创建 mock 数据
    """
    logger.info("=== 创建 Mock 数据 ===")

    feed_dict = {}

    for name, info in placeholders.items():
        shape = info["shape"]
        dtype = info["dtype"]

        # 处理形状，将 None 替换为 batch_size
        mock_shape = []
        for dim in shape:
            if dim is None:
                mock_shape.append(batch_size)
            elif dim == 0:
                # 维度为0的情况，保持为0
                mock_shape.append(0)
            else:
                mock_shape.append(dim)

        # 如果形状为空列表，说明是标量，但某些 Placeholder 可能需要特定形状
        # 检查名称中是否包含特定模式来推断正确的形状
        if not mock_shape:
            # 对于 ReadVariableOp/resource 类型的 Placeholder，尝试从名称推断形状
            if "/ReadVariableOp/resource" in name:
                # 这些通常是权重或偏置参数，需要根据上下文推断形状
                # 暂时使用一个合理的默认形状
                if "BiasAdd" in name:
                    # 偏置通常是一维向量
                    mock_shape = [32]  # 默认偏置大小
                elif "MatMul" in name or "Tensordot" in name:
                    # 权重矩阵通常是二维
                    mock_shape = [64, 32]  # 默认权重矩阵大小
                else:
                    # 其他情况使用标量
                    mock_shape = []
            else:
                # 其他标量 Placeholder 保持标量
                mock_shape = []

        # 生成 mock 数据
        if dtype == np.float32:
            # 生成随机浮点数据
            mock_data = np.random.normal(0.0, 1.0, mock_shape).astype(dtype)
        elif dtype == np.int32:
            # 生成随机整数数据
            mock_data = np.random.randint(0, 100, mock_shape).astype(dtype)
        elif dtype == np.int64:
            # 生成随机长整数数据
            mock_data = np.random.randint(0, 100, mock_shape).astype(dtype)
        elif dtype == np.bool_:
            # 生成随机布尔数据
            mock_data = np.random.choice([True, False], mock_shape).astype(dtype)
        else:
            # 默认生成浮点数据
            mock_data = np.random.normal(0.0, 1.0, mock_shape).astype(np.float32)

        feed_dict[name + ":0"] = mock_data

        # logger.info(f"Mock 数据 - {name}:")
        # logger.info(f"  形状: {mock_shape}")
        # logger.info(f"  数据类型: {dtype}")
        # logger.info(f"  数据范围: [{np.min(mock_data):.4f}, {np.max(mock_data):.4f}]")

    return feed_dict


# ==========================================
# 4. 执行推理
# ==========================================
def run_inference(graph_def, feed_dict,
                  output_node_name, config,
                  device="cpu", xla=False, num_runs=100,
                  warmup_runs=10, logger=None, silent=False):
    if not silent:
        logger.info(f"=== 执行图推理 ===")
        logger.info(f"输出节点: {output_node_name}")
        logger.info(f"设备: {device.upper()}")
        if device.lower() == "cuda":
            logger.info(f"XLA: {xla}")
        logger.info(f"预热次数: {warmup_runs}, 正式运行次数: {num_runs}")

    with tf.Graph().as_default() as graph:
        # 测量图导入时间
        t_import_start = time.time()
        tf.import_graph_def(graph_def, name="")
        t_import_end = time.time()
        # logger.info(f"[时间] 图导入耗时: {(t_import_end - t_import_start)*1000:.2f} ms")

        # 准备 Session Feed
        t_feed_start = time.time()
        session_feed_dict = {}
        for name, data in feed_dict.items():
            try:
                tensor = graph.get_tensor_by_name(name)
                session_feed_dict[tensor] = data
            except KeyError:
                pass  # 忽略图中不存在的 tensor

        # 获取输出 Tensor
        try:
            output_tensor = graph.get_tensor_by_name(f"{output_node_name}:0")
        except KeyError:
            logger.error(f"错误: 找不到输出张量 {output_node_name}:0")
            # 尝试打印所有节点找名字
            # for n in graph.as_graph_def().node: log_info(n.name)
            return None
        t_feed_end = time.time()
        # logger.info(f"[时间] Feed Dict 准备耗时: {(t_feed_end - t_feed_start)*1000:.2f} ms")


        # 测量 Session 创建时间
        t_sess_start = time.time()
        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            t_sess_end = time.time()
            # logger.info(f"[时间] Session 创建耗时: {(t_sess_end - t_sess_start)*1000:.2f} ms")

            try:
                # 预热运行
                if not silent: logger.info(f">>> 预热运行 {warmup_runs} 次...")
                for _ in range(warmup_runs):
                    _ = sess.run(output_tensor, feed_dict=session_feed_dict)
                if not silent: logger.info(">>> 预热完成")

                # 正式测量
                if not silent: logger.info(f">>> 正式运行 {num_runs} 次...")
                run_times = []
                for i in range(num_runs):
                    t_run_start = time.time()
                    result = sess.run(output_tensor, feed_dict=session_feed_dict)
                    t_run_end = time.time()
                    run_times.append((t_run_end - t_run_start) * 1000)  # ms

                batches = result.shape[0]
                if not silent:
                    print_perf_result(batches, run_times, num_runs, logger)

                return result

            except Exception as e:
                logger.error(f"!!!! 推理失败 !!!!")
                logger.error(f"错误信息: {e}")

                return None


# ==========================================
# 主函数
# ==========================================
def main():
    # 设置环境
    setup_environment()
    # 设置随机种子
    set_random_seeds()

    model_path = "./graph_def.pb"
    model_name = "prunedGraph"
    output_node_name = "predicts"  # 输出节点名称

    # 创建prunedGraph特有的参数解析器
    custom_parser = create_pruned_graph_parser()
    # 解析参数（合并基础参数和自定义参数）
    args = parse_arguments(parent_parser=custom_parser)

    # 使用模型名称设置logger
    logger = setup_logger(model_name=model_name)

    logger.info("="*50)
    logger.info("参数配置")
    logger.info("="*50)
    logger.info(f"  设备:     {args.device.upper()}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  XLA:      {args.xla if args.device == 'cuda' else 'N/A'}")
    logger.info(f"  运行次数: {args.infer_iters}")
    logger.info(f"  预热次数: {args.warmup_iters}")
    logger.info(f"  模型路径: {model_path}")
    logger.info(f"  输出节点: {output_node_name}")
    logger.info(f"  精度比对: {'是' if args.check_acc else '否'}")
    if args.check_acc:
        logger.info(f"  atol:     {args.atol}")
        logger.info(f"  rtol:     {args.rtol}")
        logger.info(f"  cossim:   {args.cossim}")
    logger.info("="*50)

    total_start = time.time()

    # 仅当 device=musa 时加载 MUSA 插件
    if args.device.lower() == "musa":
        load_musa_plugin(args.musa_plugin)

    config = create_session_config(device_type=args.device, xla=args.xla,
                                   log_device_placement=False, allow_soft_placement=False,
                                   logger=logger)

    # 1. 分析图
    t0 = time.time()
    graph_def, placeholders = load_graph_and_get_placeholders(model_path, logger)
    # logger.info(f"[时间] 图加载与分析耗时: {(time.time() - t0)*1000:.2f} ms")
    if not placeholders:
        logger.info("错误: 未找到 Placeholder")
        return

    # 2. 造数据 (含自动修复)
    t1 = time.time()
    feed_dict = create_mock_data(placeholders, args.batch_size, logger)
    # logger.info(f"[时间] Mock 数据创建耗时: {(time.time() - t1)*1000:.2f} ms")

    # 3. 跑推理
    device_result = run_inference(
        graph_def,
        feed_dict,
        output_node_name,
        config=config,
        device=args.device,
        xla=args.xla,
        num_runs=args.infer_iters,
        warmup_runs=args.warmup_iters,
        logger=logger
    )

    # 4. 如果需要，与CPU结果进行精度比对
    if args.check_acc and args.device.lower() != "cpu":

        # logger.info("\n" + "="*50)
        # logger.info("开始与CPU进行精度比对...")
        # logger.info("="*50)

        # 使用CPU运行相同的数据
        cpu_config = create_session_config(
            device_type="cpu",
            log_device_placement=False,
            allow_soft_placement=False,
            logger=logger
        )

        cpu_result = run_inference(
            graph_def,
            feed_dict,
            output_node_name,
            config=cpu_config,
            device="cpu",
            xla=False,
            num_runs=1,  # CPU只运行一次用于比对
            warmup_runs=0,
            logger=logger,
            silent=True  # CPU运行时不打印日志
        )

        if cpu_result is not None:
            # 进行精度比对
            compare_accuracy(
                cpu_result=cpu_result,
                device_result=device_result,
                atol=args.atol,
                rtol=args.rtol,
                cossim=args.cossim,
                logger=logger
            )
        else:
            logger.error("CPU推理失败，无法进行精度比对")

    total_end = time.time()
    # logger.info(f"[总耗时] {(total_end - total_start)*1000:.2f} ms")


if __name__ == "__main__":
    main()
