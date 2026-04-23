"""
公共测试工具函数

所有 test_tf_musa_extension.py 文件共用的工具函数。
"""

import argparse
import os
import random
import tensorflow as tf


def setup_environment():
    """设置环境变量"""
    musa_visible_devices = os.environ.get("MUSA_VISIBLE_DEVICES", "")
    if musa_visible_devices:
        os.environ["MUSA_VISIBLE_DEVICES"] = musa_visible_devices
        print(f"MUSA_VISIBLE_DEVICES set to: {musa_visible_devices}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Test TensorFlow MUSA Library")
    parser.add_argument(
        "--musa-plugin",
        nargs="?",
        help="Path to the TensorFlow MUSA library .so file ",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    args = parser.parse_args()
    return args


def load_musa_plugin(plugin_path):
    """加载 TensorFlow MUSA 插件"""
    tf.load_library(plugin_path)


def set_random_seeds(seed=42):
    """设置随机种子"""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)