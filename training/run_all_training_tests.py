#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一模型训练测试入口脚本

对 training 目录下所有包含 test_tf_musa_extension.py 的模型进行训练测试。
支持命令行指定训练轮数、指定GPU设备，每个模型的错误将保存到独立的日志文件中。

使用示例:
    python run_all_training_tests.py --epochs 10
    python run_all_training_tests.py --epochs 5 --log-dir error_logs
    python run_all_training_tests.py --epochs 10 --models deepfm wukong
    python run_all_training_tests.py --epochs 10 --gpu 0
    python run_all_training_tests.py --epochs 10 --gpu 0,1
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

# 模型目录列表（完整列表）
MODEL_DIRS = [
    "deepfm",
    "dien",
    "din",
    "dsin",
    "esmm",
    "fgcnn",
    "flen",
    "fwfm",
    "mmoe",
    "onetrans",
    "ple",
    "rankmixer",
    "tokenmixer-large",
    "wukong",
    "xdeepfm",
]

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent.resolve()


def discover_model_tests() -> List[str]:
    """自动发现包含 test_tf_musa_extension.py 的模型目录"""
    models_with_tests = []
    for model_dir in MODEL_DIRS:
        test_file = SCRIPT_DIR / model_dir / "test_tf_musa_extension.py"
        if test_file.exists():
            models_with_tests.append(model_dir)
    return models_with_tests


def run_model_test(
    model_name: str,
    epochs: int,
    musa_plugin: str,
    gpu_devices: str,
    log_dir: Path,
) -> Tuple[bool, str]:
    """运行单个模型的测试脚本

    Args:
        model_name: 模型名称
        epochs: 训练轮数
        musa_plugin: TensorFlow MUSA 库路径
        gpu_devices: GPU设备ID字符串 (如 "0" 或 "0,1")
        log_dir: 日志目录路径

    Returns:
        (success, output) 测试是否成功及输出信息
    """
    test_file = SCRIPT_DIR / model_name / "test_tf_musa_extension.py"

    # 设置环境变量传递 epochs 和 gpu 参数
    env = os.environ.copy()
    env["TRAIN_EPOCHS"] = str(epochs)
    if gpu_devices:
        env["MUSA_VISIBLE_DEVICES"] = gpu_devices

    # 构建命令
    cmd = [sys.executable, str(test_file)]
    if musa_plugin:
        # 将相对路径转换为绝对路径，因为子进程的工作目录在模型子目录中
        plugin_path = Path(musa_plugin)
        if not plugin_path.is_absolute():
            plugin_path = (SCRIPT_DIR / musa_plugin).resolve()
        cmd.extend(["--musa_plugin", str(plugin_path)])

    print(f"\n{'='*60}")
    print(f"Running test for model: {model_name}")
    print(f"Epochs: {epochs}")
    if gpu_devices:
        print(f"GPU Devices: {gpu_devices}")
    print(f"{'='*60}\n")

    # 为此模型创建独立的错误日志文件
    error_log_path = log_dir / f"{model_name}_error.log"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_DIR / model_name),
            env=env,
            timeout=300,  # 5分钟超时
        )

        output = result.stdout
        error_output = result.stderr

        # 打印输出
        if output:
            print(output)
        if error_output:
            print(f"[STDERR] {error_output}")

        if result.returncode != 0:
            # 将错误写入模型专属日志文件
            error_msg = f"Model: {model_name} | Return code: {result.returncode} | Error: {error_output or output}"
            with open(error_log_path, "w", encoding="utf-8") as f:
                f.write(f"{'='*60}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Return Code: {result.returncode}\n")
                f.write(f"{'='*60}\n\n")
                f.write("=== STDOUT ===\n")
                f.write(output or "(empty)\n")
                f.write("\n=== STDERR ===\n")
                f.write(error_output or "(empty)\n")
            return False, error_msg

        return True, output

    except subprocess.TimeoutExpired:
        error_msg = f"Model: {model_name} | Test timeout (>300s)"
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write(f"{'='*60}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error: Test timeout (>300s)\n")
            f.write(f"{'='*60}\n")
        print(f"[ERROR] {error_msg}")
        return False, error_msg

    except Exception as e:
        error_msg = f"Model: {model_name} | Unexpected error: {str(e)}"
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write(f"{'='*60}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error: Unexpected error - {str(e)}\n")
            f.write(f"{'='*60}\n")
        print(f"[ERROR] {error_msg}")
        return False, error_msg


def main():
    parser = argparse.ArgumentParser(
        description="统一模型训练测试入口脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python run_all_training_tests.py --epochs 10
    python run_all_training_tests.py --epochs 5 --log-dir error_logs
    python run_all_training_tests.py --epochs 10 --models deepfm wukong dien
    python run_all_training_tests.py --epochs 10 --gpu 0
    python run_all_training_tests.py --epochs 10 --gpu 0,1,2
    python run_all_training_tests.py --list-models
        """
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="训练轮数 (默认: 10)",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="error_logs",
        help="错误日志目录名 (默认: error_logs)，每个模型的错误将保存到独立的文件中",
    )

    parser.add_argument(
        "--models",
        nargs="*",
        help="指定要测试的模型列表 (默认: 所有模型)",
    )

    parser.add_argument(
        "--musa-plugin",
        type=str,
        default=None,
        help="TensorFlow MUSA 库 .so 文件路径",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出所有可测试的模型",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="并行运行测试 (注意: 可能导致 GPU 资源竞争)",
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="指定GPU设备ID (如: 0 或 0,1,2)",
    )

    args = parser.parse_args()

    # 发现可用模型
    available_models = discover_model_tests()

    if args.list_models:
        print("Available models with test_tf_musa_extension.py:")
        for model in available_models:
            print(f"  - {model}")
        print(f"\nTotal: {len(available_models)} models")
        return 0

    # 确定要测试的模型
    models_to_test = args.models if args.models else available_models

    # 验证模型存在
    invalid_models = [m for m in models_to_test if m not in available_models]
    if invalid_models:
        print(f"[ERROR] Invalid models: {invalid_models}")
        print(f"Available models: {available_models}")
        return 1

    # 创建错误日志目录
    log_dir = SCRIPT_DIR / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# 开始模型训练测试")
    print(f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# 训练轮数: {args.epochs}")
    if args.gpu:
        print(f"# GPU设备: {args.gpu}")
    print(f"# 测试模型: {models_to_test}")
    print(f"# 错误日志目录: {args.log_dir}/<model_name>_error.log")
    print(f"{'#'*60}\n")

    # 运行测试
    results: Dict[str, Tuple[bool, str]] = {}

    for model_name in models_to_test:
        success, output = run_model_test(
            model_name,
            args.epochs,
            args.musa_plugin,
            args.gpu,
            log_dir,
        )
        results[model_name] = (success, output)

    # 输出测试结果摘要
    print(f"\n{'#'*60}")
    print(f"# 测试结果摘要")
    print(f"{'#'*60}\n")

    success_count = 0
    failed_count = 0

    for model_name, (success, _) in results.items():
        status = "PASSED" if success else "FAILED"
        symbol = "[OK]" if success else "[FAIL]"
        print(f"{symbol} {model_name}: {status}")
        if success:
            success_count += 1
        else:
            failed_count += 1

    print(f"\n总计: {len(results)} 个模型")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")

    if failed_count > 0:
        print(f"\n错误详情请查看日志目录: {args.log_dir}/")
        for model_name, (success, _) in results.items():
            if not success:
                print(f"  - {model_name}: {args.log_dir}/{model_name}_error.log")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())