import os
import sys
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_utils import (
    setup_environment,
    parse_arguments,
    load_musa_plugin,
    set_random_seeds,
)
from model.rankmixer import RankMixer
from data.criteo_kaggle_dataset import get_dataset


def create_model():
    """创建 RankMixer 模型"""
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
    NUM_TOKENS = 16
    NUM_HEADS = 16
    EXPANSION_RATIO = 4
    NUM_HIDDEN_HEAD = 2
    DIM_HIDDEN_HEAD = 256
    DROPOUT = 0.5
    BIAS = False

    model = RankMixer(
        num_layers=NUM_LAYERS,
        num_sparse_embs=NUM_SPARSE_EMBS,
        num_tokens=NUM_TOKENS,
        dim_input_sparse=NUM_CAT_FEATURES,
        dim_input_dense=NUM_DENSE_FEATURES,
        dim_emb=DIM_EMB,
        num_heads=NUM_HEADS,
        expansion_ratio=EXPANSION_RATIO,
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
    embedding_optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)
    other_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
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

    accuracy = float(num_correct / num_samples) if num_samples > 0 else 0.0
    recall_pos = float(pos_correct / pos_samples) if pos_samples > 0 else 0.0
    return accuracy, num_samples, recall_pos, pos_samples


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
    embedding_optimizer, other_optimizer, criterion = create_optimizer_and_criterion()

    # 创建数据集
    train_dataset, valid_dataset = create_dataset()

    try:
        # 训练循环：执行 epochs 次训练步骤
        for epoch in range(epochs):
            for inputs, labels in train_dataset:
                loss = train_step(
                    model,
                    inputs,
                    labels,
                    embedding_optimizer,
                    other_optimizer,
                    criterion,
                )
                assert not tf.math.is_nan(loss), "Loss is NaN, stopping training."
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy():.4f}")
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
