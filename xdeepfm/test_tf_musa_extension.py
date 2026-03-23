import argparse
import os
import random
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tf_test_model.utils import get_default_musa_plugin_path, resolve_musa_plugin_path

parser = argparse.ArgumentParser(description="Test TensorFlow MUSA Library")
parser.add_argument(
    "tensorflow_musa_library_path",
    nargs="?",
    default=get_default_musa_plugin_path(),
    help="Path to the TensorFlow MUSA library .so file "
    "(defaults to the auto-detected sibling build path)",
)
args = parser.parse_args()
plugin_path = resolve_musa_plugin_path(args.tensorflow_musa_library_path)

import tensorflow as tf

tf.load_library(plugin_path)

import numpy as np

from model.xdeepfm import XDeepFM
from model.lr_schedule import LinearWarmup

# Example sibling-workspace path (preferred):
#   ../tensorflow_musa_extension/build/libmusa_plugin.so
# Example docker absolute path (fallback):
#   /workspace/tensorflow_musa_extension/build/libmusa_plugin.so


####################################################################################################
#                                           SET RANDOM SEEDS                                       #
####################################################################################################
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

####################################################################################################
#                                  DATASET SPECIFIC CONFIGURATION                                  #
####################################################################################################
NUM_CAT_FEATURES = 26
NUM_DENSE_FEATURES = 13
NUM_SPARSE_EMBS = [1000] * NUM_CAT_FEATURES
DIM_OUTPUT = 1

####################################################################################################
#                                   MODEL SPECIFIC CONFIGURATION                                   #
####################################################################################################
DIM_EMB = 128
CIN_LAYER_SIZE = (128, 128)
CIN_SPLIT_HALF = True
CIN_ACTIVATION = None
L2_REG_LINEAR = 1e-6
L2_REG_CIN = 1e-6
L2_REG_DNN = 1e-6
DNN_HIDDEN_UNITS = (256, 64)
DROPOUT = 0.5
BIAS = False

####################################################################################################
#                                           CREATE MODEL                                           #
####################################################################################################
model = XDeepFM(
    num_sparse_embs=NUM_SPARSE_EMBS,
    dim_emb=DIM_EMB,
    dim_input_sparse=NUM_CAT_FEATURES,
    dim_input_dense=NUM_DENSE_FEATURES,
    cin_layer_size=CIN_LAYER_SIZE,
    cin_split_half=CIN_SPLIT_HALF,
    cin_activation=CIN_ACTIVATION,
    l2_reg_linear=L2_REG_LINEAR,
    l2_reg_cin=L2_REG_CIN,
    l2_reg_dnn=L2_REG_DNN,
    dnn_hidden_units=DNN_HIDDEN_UNITS,
    dnn_dropout=DROPOUT,
    dim_output=DIM_OUTPUT,
    bias=BIAS,
    seed=SEED,
)

####################################################################################################
#                                  TRAINING SPECIFIC CONFIGURATION                                 #
####################################################################################################
BATCH_SIZE = 2
TRAIN_EPOCHS = 10
PEAK_LR = 0.004
INIT_LR = 1e-8
TOTAL_STEPS_PER_EPOCH = 39291958 // BATCH_SIZE
TOTAL_ITERS = TOTAL_STEPS_PER_EPOCH

lr_schedule = LinearWarmup(
    initial_learning_rate=INIT_LR, peak_learning_rate=PEAK_LR, warmup_steps=TOTAL_ITERS
)
embedding_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
other_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
criterion = tf.keras.losses.BinaryCrossentropy(from_logits=False)

####################################################################################################
#                                    BUILD MODEL & SEPARATE VARS                                   #
####################################################################################################
# TF Lazy Execution
dummy_sparse = tf.zeros((1, NUM_CAT_FEATURES), dtype=tf.int32)
dummy_dense = tf.zeros((1, NUM_DENSE_FEATURES), dtype=tf.float32)
_ = model((dummy_sparse, dummy_dense))

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


####################################################################################################
#                                          VALID FUNCTION                                          #
####################################################################################################
def validate(model, dataset):
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


####################################################################################################
#                                         TRAINING STEP                                            #
####################################################################################################
@tf.function
def train_step(inputs, labels):
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


####################################################################################################
#                                           TRAINING LOOP                                          #
####################################################################################################

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

try:
    _ = train_step(inputs, labels)
    _ = validate(model, [(inputs, labels)])
except Exception as e:
    print(f"Error during training or validation: {e}")
    sys.exit(1)
print(f"{args.tensorflow_musa_library_path} test passed!")
