import os
import sys
import tensorflow as tf
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_utils import setup_environment, parse_arguments, load_musa_plugin, set_random_seeds
from model.mmoe import MMoE
from model.lr_schedule import LinearWarmup


def create_model():
    """创建 MMOE 模型"""
    # 配置参数
    NUM_CAT_FEATURES = 26
    NUM_DENSE_FEATURES = 13
    NUM_SPARSE_EMBS = [1000] * NUM_CAT_FEATURES
    DIM_OUTPUT = 1

    DIM_EMB = 64
    NUM_TASKS = 2
    TASK_NAMES = ["ctr", "aux"]
    NUM_EXPERTS = 8
    EXPERT_HIDDEN_UNITS = (256, 128)
    TOWER_HIDDEN_UNITS = (64,)
    ACTIVATION = "relu"
    DROPOUT = 0.0
    USE_BIAS = False

    model = MMoE(
        num_sparse_embs=NUM_SPARSE_EMBS,
        dim_emb=DIM_EMB,
        dim_input_sparse=NUM_CAT_FEATURES,
        dim_input_dense=NUM_DENSE_FEATURES,
        num_tasks=NUM_TASKS,
        task_names=TASK_NAMES,
        num_experts=NUM_EXPERTS,
        expert_hidden_units=EXPERT_HIDDEN_UNITS,
        tower_hidden_units=TOWER_HIDDEN_UNITS,
        activation=ACTIVATION,
        dropout=DROPOUT,
        use_bias=USE_BIAS,
    )

    return model


def separate_trainable_variables(model):
    """分离可训练变量"""
    # 配置参数
    NUM_CAT_FEATURES = 26
    NUM_DENSE_FEATURES = 13

    # 构建模型
    dummy_static_sparse = tf.zeros((1, NUM_CAT_FEATURES), dtype=tf.int32)
    dummy_dense = tf.zeros((1, NUM_DENSE_FEATURES), dtype=tf.float32)
    dummy_inputs = (dummy_static_sparse, dummy_dense)
    _ = model(dummy_inputs, training=False)

    embedding_parameters = []
    other_parameters = []

    for var in model.trainable_variables:
        if hasattr(var, "path"):
            # path is available in TF 2.13+
            if "sparse_embedding" in var.path and "embeddings" in var.name:
                embedding_parameters.append(var)
            else:
                other_parameters.append(var)
        else:
            if "sparse_embedding" in var.name:
                embedding_parameters.append(var)
            else:
                other_parameters.append(var)

    return embedding_parameters, other_parameters


def create_optimizer_and_criterion():
    """创建优化器和损失函数"""
    BATCH_SIZE = 2
    PEAK_LR = 0.001
    INIT_LR = 1e-8
    TOTAL_STEPS_PER_EPOCH = 39291958 // BATCH_SIZE
    TOTAL_ITERS = TOTAL_STEPS_PER_EPOCH

    lr_schedule = LinearWarmup(
        initial_learning_rate=INIT_LR, peak_learning_rate=PEAK_LR, warmup_steps=TOTAL_ITERS
    )
    embedding_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    other_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    task_criteria = [
        tf.keras.losses.BinaryCrossentropy(from_logits=False),
        tf.keras.losses.BinaryCrossentropy(from_logits=False),
    ]
    LOSS_WEIGHTS = [1.0, 1.0]

    return embedding_optimizer, other_optimizer, task_criteria, LOSS_WEIGHTS


def create_sample_data():
    """创建示例数据"""
    inputs = (
        tf.convert_to_tensor(
            np.array(
                [
                    [
                        0,
                        101,
                        110,
                        239,
                        3,
                        5,
                        106,
                        5,
                        0,
                        284,
                        101,
                        104,
                        99,
                        0,
                        406,
                        260,
                        1,
                        291,
                        1,
                        2,
                        992,
                        0,
                        1,
                        187,
                        1,
                        2,
                    ],
                    [
                        22,
                        67,
                        130,
                        111,
                        0,
                        1,
                        220,
                        5,
                        0,
                        296,
                        64,
                        123,
                        63,
                        1,
                        124,
                        120,
                        0,
                        101,
                        1,
                        2,
                        123,
                        0,
                        0,
                        4,
                        1,
                        2,
                    ],
                ],
                dtype=np.int32,
            )
        ),
        tf.convert_to_tensor(
            np.array(
                [
                    [
                        1.0986123,
                        5.1474943,
                        1.0986123,
                        2.0794415,
                        3.0445225,
                        2.0794415,
                        1.0986123,
                        2.0794415,
                        2.0794415,
                        0.6931472,
                        0.6931472,
                        0.0,
                        2.0794415,
                    ],
                    [
                        0.0,
                        5.4595857,
                        0.6931472,
                        0.6931472,
                        8.016977,
                        5.1474943,
                        4.158883,
                        2.3978953,
                        5.170484,
                        0.0,
                        2.0794415,
                        0.0,
                        0.6931472,
                    ],
                ],
                dtype=np.float32,
            )
        ),
    )

    labels = tf.convert_to_tensor(np.array([1, 0], dtype=np.float32))

    return inputs, labels


def get_task2_label(dense_inputs):
    """获取任务2标签"""
    return tf.cast(dense_inputs[:, 0] > 0.0, tf.float32)


def validate(model, dataset):
    """验证函数"""
    num_samples = 0
    num_correct = 0
    pos_samples = 0
    pos_correct = 0

    for inputs, labels in dataset:
        outputs_list = model(inputs, training=False)
        outputs = tf.squeeze(outputs_list[0])

        labels = tf.cast(labels, tf.float32)

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


@tf.function
def train_step(model, inputs, labels, embedding_optimizer, other_optimizer, task_criteria, LOSS_WEIGHTS):
    """单步训练函数"""
    _, dense_inputs = inputs
    labels_task1 = tf.cast(labels, tf.float32)
    labels_task2 = get_task2_label(dense_inputs)

    with tf.GradientTape() as tape:
        logits_list = model(inputs, training=True)
        loss1 = task_criteria[0](labels_task1, tf.squeeze(logits_list[0]))
        loss2 = task_criteria[1](labels_task2, tf.squeeze(logits_list[1]))
        loss = LOSS_WEIGHTS[0] * loss1 + LOSS_WEIGHTS[1] * loss2

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

    # 分离训练变量
    embedding_parameters, other_parameters = separate_trainable_variables(model)

    # 创建优化器和损失函数
    embedding_optimizer, other_optimizer, task_criteria, LOSS_WEIGHTS = create_optimizer_and_criterion()

    # 创建示例数据
    inputs, labels = create_sample_data()

    try:
        # 训练循环：执行 epochs 次训练步骤
        for epoch in range(epochs):
            loss = train_step(model, inputs, labels, embedding_optimizer, other_optimizer, task_criteria, LOSS_WEIGHTS)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy():.4f}")
        _ = validate(model, [(inputs, labels)])
    except Exception as e:
        print(f"Error during training or validation: {e}")
        sys.exit(1)
    print(f"{plugin_path} test passed!")


if __name__ == "__main__":
    main()