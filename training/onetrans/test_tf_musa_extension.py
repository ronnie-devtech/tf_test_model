import os
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
from model.onetrans import OneTrans
from data.criteo_kaggle_dataset import get_dataset


def create_model():
    """创建 OneTrans 模型"""
    # 配置参数
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

    NUM_LAYERS = 3
    LS = 26
    LNS = 13
    DIM_EMB = 128
    NUM_HEADS = 16
    D_FF = 512
    NUM_HIDDEN_HEAD = 2
    DIM_HIDDEN_HEAD = 256

    model = OneTrans(
        num_layers=NUM_LAYERS,
        LS=LS,
        LNS=LNS,
        dim_emb=DIM_EMB,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_sparse_embs=NUM_SPARSE_EMBS,
        num_hidden_head=NUM_HIDDEN_HEAD,
        dim_hidden_head=DIM_HIDDEN_HEAD,
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
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)
    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    return optimizer, criterion


def validate(model, dataset, criterion):
    num_samples = 0
    total_logloss = 0.0
    num_batches = 0
    auc_metric = tf.keras.metrics.AUC(from_logits=True, name="val_auc")

    for inputs, labels in dataset:
        outputs = model(inputs, training=False)
        labels = tf.cast(labels, tf.float32)
        outputs = tf.squeeze(outputs)

        # Calculate batch Logloss
        batch_loss = criterion(labels, outputs)
        total_logloss += batch_loss.numpy()
        num_batches += 1

        # Update AUC metric
        auc_metric.update_state(labels, outputs)
        num_samples += labels.shape[0]

    # Compute final metrics
    avg_logloss = float(total_logloss / num_batches) if num_batches > 0 else 0.0
    auc_score = float(auc_metric.result().numpy())
    auc_metric.reset_states()
    return avg_logloss, auc_score, num_samples


def get_batch_size(labels):
    """获取当前 batch size，兼容动态图 shape。"""
    if labels.shape[0] is not None:
        return int(labels.shape[0])
    return int(tf.shape(labels)[0].numpy())


@tf.function
def train_step(model, inputs, labels, optimizer, criterion):
    """单步训练函数"""
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = criterion(labels, tf.squeeze(outputs))

    grads = tape.gradient(loss, model.trainable_variables)
    grads_and_vars = []
    for grad, var in zip(grads, model.trainable_variables):
        if grad is not None:
            grads_and_vars.append((grad, var))
    optimizer.apply_gradients(grads_and_vars)

    return loss


def main():
    """主函数"""
    # 设置环境
    setup_environment()

    # 解析参数
    args = parse_arguments()
    plugin_path = args.musa_plugin
    epochs = args.epochs

    # 加载 MUSA 插件
    load_musa_plugin(plugin_path)

    # 设置随机种子
    set_random_seeds()

    # 创建模型
    model = create_model()

    # 创建优化器和损失函数
    optimizer, criterion = create_optimizer_and_criterion()

    # 创建数据集
    train_dataset, valid_dataset = create_dataset()

    try:
        # 训练循环：执行 epochs 次训练步骤
        global_step = 0
        for epoch in range(epochs):
            prev_loss_value = None
            epoch_loss_sum = 0.0
            epoch_step_count = 0
            for inputs, labels in train_dataset:
                global_step += 1
                epoch_step_count += 1
                iter_start = time.perf_counter()
                loss = train_step(model, inputs, labels, optimizer, criterion)
                assert not tf.math.is_nan(loss), "Loss is NaN, stopping training."
                loss_value = float(loss.numpy())
                iter_time_sec = time.perf_counter() - iter_start
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
            val_logloss, val_auc, val_samples = validate(
                model, valid_dataset, criterion
            )
            print(
                f"Val Logloss: {val_logloss:.4f}, "
                f"Val AUC: {val_auc:.4f}, "
                f"Total Val Samples: {val_samples}"
            )
    except Exception as e:
        print(f"Error during training or validation: {e}")
        sys.exit(1)
    print(f"{plugin_path} test passed!")


if __name__ == "__main__":
    main()
