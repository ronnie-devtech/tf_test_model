import os


def _env_flag(name):
    value = os.environ.get(name, "")
    return value.lower() in ("1", "true", "on", "yes")


def _parse_cpu_affinity(spec):
    cpus = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            start = int(start)
            end = int(end)
            if end < start:
                raise ValueError(f"invalid CPU range: {part}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        raise ValueError("empty CPU affinity")
    return cpus


def _configure_fast_runtime_preset():
    """Configure measured-fast S5000 runtime knobs before TensorFlow import."""
    if not _env_flag("TOKENMIXER_FAST_RUNTIME"):
        return []

    defaults = {
        "TF_NUM_INTEROP_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "OMP_NUM_THREADS": "1",
        "KMP_BLOCKTIME": "0",
        "MUSA_PAGEABLE_H2D_ON_COMPUTE_STREAM": "1",
        "TF_CPP_MIN_LOG_LEVEL": "2",
    }
    applied = []
    for name, value in defaults.items():
        before = os.environ.get(name)
        os.environ.setdefault(name, value)
        after = os.environ.get(name)
        if before is None:
            applied.append(f"{name}={after}")

    affinity = os.environ.get("TOKENMIXER_CPU_AFFINITY", "0-7")
    if affinity.lower() not in ("", "none", "off"):
        try:
            cpus = _parse_cpu_affinity(affinity)
            os.sched_setaffinity(0, cpus)
            applied.append(f"cpu_affinity={affinity}")
        except Exception as exc:
            applied.append(f"cpu_affinity={affinity} failed: {exc}")

    return applied


_FAST_RUNTIME_PRESET = _configure_fast_runtime_preset()

import sys
import time
import tensorflow as tf
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_utils import (
    setup_environment,
    parse_arguments,
    load_musa_plugin,
    set_random_seeds,
)
from model.tokenmixerlarge import TokenMixerLarge
from data.criteo_kaggle_dataset import get_dataset


def create_model():
    """创建 TokenMixerLarge 模型"""
    # 配置参数
    NUM_CAT_FEATURES = 26
    NUM_DENSE_FEATURES = 13
    NUM_SPARSE_EMBS = [
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

    DIM_OUTPUT = 1

    NUM_LAYERS = 6
    DIM_EMB = 128
    NUM_HEADS = 16
    NUM_EXPERTS = 8
    TOP_K = 4
    NUM_HIDDEN_HEAD = 2
    DIM_HIDDEN_HEAD = 256
    DROPOUT = 0.5
    BIAS = True

    model = TokenMixerLarge(
        group_dims=[[DIM_EMB] * NUM_CAT_FEATURES, [DIM_EMB] * NUM_DENSE_FEATURES],
        num_layers=NUM_LAYERS,
        num_sparse_embs=NUM_SPARSE_EMBS,
        dim_input_sparse=NUM_CAT_FEATURES,
        dim_input_dense=NUM_DENSE_FEATURES,
        dim_emb=DIM_EMB,
        num_heads=NUM_HEADS,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        num_hidden_head=NUM_HIDDEN_HEAD,
        dim_hidden_head=DIM_HIDDEN_HEAD,
        dim_output=DIM_OUTPUT,
        dropout=DROPOUT,
        bias=BIAS,
    )

    return model


def create_dataset():
    """创建数据集"""
    NPZ_FILE_PATH = os.path.join(
        os.path.dirname(__file__), "data/kaggleAdDisplayChallenge_processed_sub.npz"
    )
    BATCH_SIZE = 4096
    train_dataset = get_dataset(
        npz_file_path=NPZ_FILE_PATH,
        split="train",
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    valid_dataset = get_dataset(
        npz_file_path=NPZ_FILE_PATH,
        split="valid",
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return train_dataset, valid_dataset


def create_optimizer_and_criterion():
    """创建优化器和损失函数"""

    embedding_optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    other_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    return embedding_optimizer, other_optimizer, criterion


def validate(model, dataset):
    """验证函数"""
    num_samples = 0
    num_correct = 0
    pos_samples = 0
    pos_correct = 0

    for inputs, labels in dataset:
        outputs = model(inputs, training=False)

        labels = tf.cast(labels, tf.float32)
        outputs = tf.squeeze(outputs)

        predictions = tf.cast(outputs >= 0.5, tf.float32)

        num_samples += labels.shape[0]
        pos_samples += tf.reduce_sum(labels).numpy()

        correct_preds = tf.cast(tf.equal(predictions, labels), tf.float32)
        num_correct += tf.reduce_sum(correct_preds).numpy()

        pos_mask = tf.equal(labels, 1.0)
        pos_correct += tf.reduce_sum(tf.boolean_mask(predictions, pos_mask)).numpy()

    accuracy = num_correct / num_samples if num_samples > 0 else 0
    recall_pos = pos_correct / pos_samples if pos_samples > 0 else 0
    return accuracy, num_samples, recall_pos, pos_samples


def get_batch_size(labels):
    """获取当前 batch size，兼容动态图 shape。"""
    if labels.shape[0] is not None:
        return int(labels.shape[0])
    return int(tf.shape(labels)[0].numpy())


@tf.function
def train_step(model, inputs, labels, embedding_optimizer, other_optimizer, criterion):
    """单步训练函数"""
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = criterion(labels, tf.squeeze(outputs))

    grads = tape.gradient(loss, model.trainable_variables)
    emb_grads = []
    other_grads = []

    for grad, var in zip(grads, model.trainable_variables):
        if grad is not None:
            if hasattr(var, "path"):
                # path is available in TF 2.13+
                if "sparse_embedding" in var.path and "embeddings" in var.name:
                    emb_grads.append((grad, var))
                else:
                    other_grads.append((grad, var))
            else:
                if "sparse_embedding" in var.name:
                    emb_grads.append((grad, var))
                else:
                    other_grads.append((grad, var))
    embedding_optimizer.apply_gradients(emb_grads)
    other_optimizer.apply_gradients(other_grads)

    return loss


def main():
    """主函数"""
    # 设置环境
    setup_environment()

    if _FAST_RUNTIME_PRESET:
        print("TOKENMIXER_FAST_RUNTIME=1")
        print("Fast runtime preset: " + ", ".join(_FAST_RUNTIME_PRESET))

    # 解析参数
    args = parse_arguments()
    plugin_path = args.musa_plugin
    epochs = args.epochs
    enable_tf32 = args.enable_tf32

    if enable_tf32:
        os.environ["MUSA_ENABLE_TF32"] = "1"
        print("MUSA_ENABLE_TF32=1")

    # 加载 MUSA 插件
    load_musa_plugin(plugin_path)

    # 设置随机种子
    set_random_seeds()

    # 创建模型
    model = create_model()

    # 创建优化器和损失函数
    embedding_optimizer, other_optimizer, criterion = create_optimizer_and_criterion()

    # 创建数据集
    train_dataset, valid_dataset = create_dataset()

    try:
        # 训练循环：执行 epochs 次训练止骤
        global_step = 0
        for epoch in range(epochs):
            prev_loss_value = None
            epoch_loss_sum = 0.0
            epoch_step_count = 0
            for inputs, labels in train_dataset:
                global_step += 1
                epoch_step_count += 1
                iter_start = time.perf_counter()
                loss = train_step(
                    model,
                    inputs,
                    labels,
                    embedding_optimizer,
                    other_optimizer,
                    criterion,
                )
                iter_time_sec = time.perf_counter() - iter_start
                assert not tf.math.is_nan(loss), "Loss is NaN, stopping training."
                loss_value = float(loss.numpy())
                batch_size = get_batch_size(labels)
                samples_per_sec = batch_size / iter_time_sec if iter_time_sec > 0 else 0.0
                epoch_loss_sum += loss_value

                if prev_loss_value is None:
                    loss_drop = None
                    loss_drop_pct = None
                    loss_drop_per_sec = None
                else:
                    loss_drop = prev_loss_value - loss_value
                    loss_drop_pct = (
                        (loss_drop / prev_loss_value) * 100.0
                        if prev_loss_value != 0
                        else 0.0
                    )
                    loss_drop_per_sec = (
                        loss_drop / iter_time_sec if iter_time_sec > 0 else 0.0
                    )

                avg_loss = epoch_loss_sum / epoch_step_count
                if loss_drop is None:
                    convergence_msg = "loss_drop=N/A, loss_drop_pct=N/A, loss_drop_per_sec=N/A"
                else:
                    convergence_msg = (
                        f"loss_drop={loss_drop:+.6f}, "
                        f"loss_drop_pct={loss_drop_pct:+.2f}%, "
                        f"loss_drop_per_sec={loss_drop_per_sec:+.6f}/s"
                    )

                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Iter {epoch_step_count}, "
                    f"Global Step {global_step}, "
                    f"Loss: {loss_value:.6f}, "
                    f"Avg Loss: {avg_loss:.6f}, "
                    f"{convergence_msg}, "
                    f"Step Time: {iter_time_sec:.4f}s, "
                    f"Samples/s: {samples_per_sec:.2f}"
                )
                prev_loss_value = loss_value

            accuracy, num_samples, recall_pos, pos_samples = validate(
                model, valid_dataset
            )
            print(
                f"Accuracy: {float(accuracy)*100:.2f}%, "
                f"Total Samples: {num_samples}, "
                f"Positive Recall: {float(recall_pos)*100:.2f}%, "
                f"Positive Samples: {pos_samples}"
            )
    except Exception as e:
        print(f"Error during training or validation: {e}")
        sys.exit(1)
    print(f"{plugin_path} test passed!")


if __name__ == "__main__":
    main()
