from typing import List
import tensorflow as tf
from tensorflow.keras import layers

from model.mlp import MLP
from model.embedding import Embedding


class RMSNorm(layers.Layer):
    def __init__(self, dim, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.dim = dim

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale", shape=(self.dim,), initializer="ones", trainable=True
        )

    def call(self, x):
        norm = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        x = x * tf.math.rsqrt(norm + self.eps)
        return self.scale * x


def _concat_glorot_initializer(split_sizes):
    glorot = tf.keras.initializers.GlorotUniform()

    def _initializer(shape, dtype=None):
        shape = tf.TensorShape(shape).as_list()
        prefix_shape = shape[:-1]
        parts = [glorot(prefix_shape + [size], dtype=dtype) for size in split_sizes]
        return tf.concat(parts, axis=-1)

    return _initializer


class PertokenSwiGLU(layers.Layer):
    def __init__(self, dim, hidden_mult=4, down_scale=0.01, bias=False, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = int(dim * hidden_mult)

        self.fc_up = layers.Dense(hidden_dim, use_bias=bias)
        self.fc_gate = layers.Dense(hidden_dim, use_bias=bias)
        self.fc_down = layers.Dense(
            dim,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=down_scale, mode="fan_avg", distribution="uniform"
            ),
            use_bias=bias,
        )

    def call(self, x):
        up = self.fc_up(x)
        gate_logits = self.fc_gate(x)
        # Swish = x * sigmoid(x)
        gate = tf.nn.sigmoid(gate_logits) * gate_logits
        return self.fc_down(up * gate)


class StackedExpertSwiGLU(layers.Layer):
    def __init__(
        self, num_experts, dim, hidden_mult=4, down_scale=0.01, bias=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.dim = dim
        self.hidden_dim = int(dim * hidden_mult)
        self.use_bias = bias
        self.down_scale = down_scale

    def build(self, input_shape):
        hidden_shape = (self.num_experts, self.dim, self.hidden_dim)
        down_shape = (self.num_experts, self.hidden_dim, self.dim)
        self.up_kernel = self.add_weight(
            name="up_kernel",
            shape=hidden_shape,
            initializer="glorot_uniform",
            trainable=True,
        )
        self.gate_kernel = self.add_weight(
            name="gate_kernel",
            shape=hidden_shape,
            initializer="glorot_uniform",
            trainable=True,
        )
        self.down_kernel = self.add_weight(
            name="down_kernel",
            shape=down_shape,
            initializer=tf.keras.initializers.VarianceScaling(
                scale=self.down_scale, mode="fan_avg", distribution="uniform"
            ),
            trainable=True,
        )
        if self.use_bias:
            self.up_bias = self.add_weight(
                name="up_bias",
                shape=(self.num_experts, self.hidden_dim),
                initializer="zeros",
                trainable=True,
            )
            self.gate_bias = self.add_weight(
                name="gate_bias",
                shape=(self.num_experts, self.hidden_dim),
                initializer="zeros",
                trainable=True,
            )
            self.down_bias = self.add_weight(
                name="down_bias",
                shape=(self.num_experts, self.dim),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.up_bias = None
            self.gate_bias = None
            self.down_bias = None

    def call(self, x):
        up = tf.einsum("btd,edh->bteh", x, self.up_kernel)
        gate_logits = tf.einsum("btd,edh->bteh", x, self.gate_kernel)
        if self.use_bias:
            up = up + self.up_bias[tf.newaxis, tf.newaxis, :, :]
            gate_logits = gate_logits + self.gate_bias[tf.newaxis, tf.newaxis, :, :]
        gate = tf.nn.sigmoid(gate_logits) * gate_logits
        hidden = up * gate
        outputs = tf.einsum("bteh,ehd->bted", hidden, self.down_kernel)
        if self.use_bias:
            outputs = outputs + self.down_bias[tf.newaxis, tf.newaxis, :, :]
        return outputs


class SparsePertokenMoE(layers.Layer):
    def __init__(
        self,
        dim,
        num_experts=4,
        top_k=2,
        hidden_mult=4,
        alpha=2.0,
        bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = alpha
        self.num_routed_experts = num_experts - 1

        self.router = layers.Dense(num_experts, use_bias=bias)
        self.experts = StackedExpertSwiGLU(
            self.num_routed_experts, dim, hidden_mult, bias=bias
        )
        self.shared_expert = PertokenSwiGLU(dim, hidden_mult, bias=bias)

    def call(self, x):
        logits = self.router(x)
        probs = tf.nn.softmax(logits, axis=-1)
        topk_vals, topk_idx = tf.math.top_k(probs, k=self.top_k)

        routed_outputs = self.experts(x)
        routed_mask = tf.one_hot(
            topk_idx[..., : self.top_k - 1],
            depth=self.num_routed_experts,
            dtype=x.dtype,
        )
        selected_outputs = tf.einsum(
            "bted,btke->btkd", routed_outputs, routed_mask
        )
        weighted_outputs = selected_outputs * topk_vals[..., : self.top_k - 1, tf.newaxis]
        output = self.alpha * tf.reduce_sum(weighted_outputs, axis=2)
        output += self.shared_expert(x)
        return output


class MixingReverting(layers.Layer):
    def __init__(self, dim, num_heads, num_tokens, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.dim = dim
        self.num_tokens = num_tokens

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        d = dim // num_heads
        mix_dim = num_tokens * d

        self.mixing = PertokenSwiGLU(mix_dim, bias=bias)
        self.reverting = PertokenSwiGLU(dim, bias=bias)

    def call(self, x):
        # B, T, D
        batch_size = tf.shape(x)[0]
        H = self.num_heads
        d = self.dim // H
        T = self.num_tokens

        x_norm = self.norm1(x)

        # Reshape for multi-head mixing: [B, T, H, d]
        x_split = tf.reshape(x_norm, (batch_size, T, H, d))
        # Permute to: [H, B, T, d]
        x_split = tf.transpose(x_split, perm=[2, 0, 1, 3])
        # Flatten T and d: [H, B, T*d]
        x_split = tf.reshape(x_split, (H, batch_size, T * d))

        x_mixed = self.mixing(x_split)

        # Reverse: [H, B, T, d]
        x_rev = tf.reshape(x_mixed, (H, batch_size, T, d))
        # [B, T, H, d]
        x_rev = tf.transpose(x_rev, perm=[1, 2, 0, 3])
        # [B, T, D]
        x_rev = tf.reshape(x_rev, (batch_size, T, self.dim))

        return x + self.norm2(self.reverting(x_rev))


class TokenMixerLargeBlock(layers.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        num_tokens,
        num_experts=4,
        top_k=2,
        hidden_mult=4,
        bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mr = MixingReverting(dim, num_heads, num_tokens, bias=bias)
        self.norm = RMSNorm(dim)
        self.moe = SparsePertokenMoE(dim, num_experts, top_k, hidden_mult, bias=bias)

    def call(self, x):
        x = self.mr(x)
        x = x + self.moe(self.norm(x))
        return x


class SemanticTokenizer(layers.Layer):
    def __init__(self, group_dims, model_dim, bias=False, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.mlps = []
        for dims in group_dims:
            self.mlps.append(
                tf.keras.Sequential(
                    [
                        layers.Dense(model_dim, activation="relu", use_bias=bias),
                        layers.Dropout(dropout),
                        layers.Dense(model_dim, use_bias=bias),
                    ]
                )
            )

        self.global_mlp = tf.keras.Sequential(
            [
                layers.Dense(model_dim, activation="relu"),
                layers.Dropout(dropout),
                layers.Dense(model_dim, use_bias=bias),
            ]
        )

    def call(self, groups):
        tokens = []
        for group_tensor, mlp in zip(groups, self.mlps):
            if isinstance(group_tensor, (list, tuple)):
                concat = tf.concat(group_tensor, axis=-1)
            else:
                batch_size = tf.shape(group_tensor)[0]
                concat = tf.reshape(group_tensor, (batch_size, -1))
            tokens.append(mlp(concat))

        stacked = tf.stack(tokens, axis=1)
        batch_size = tf.shape(stacked)[0]
        flattened = tf.reshape(stacked, (batch_size, -1))
        global_token = self.global_mlp(flattened)
        global_token = tf.expand_dims(global_token, axis=1)
        return tf.concat([global_token, stacked], axis=1)


class TokenMixerLarge(tf.keras.Model):
    def __init__(
        self,
        group_dims: List[List[int]],
        num_layers: int,
        num_sparse_embs: List[int],
        dim_input_sparse: int,
        dim_input_dense: int,
        dim_emb: int,
        num_heads: int,
        num_experts: int,
        top_k: int,
        num_hidden_head: int,
        dim_hidden_head: int,
        dim_output: int,
        dropout: float = 0.0,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding = Embedding(num_sparse_embs, dim_emb, dim_input_dense, bias)
        self.dim_emb = dim_emb
        self.dim_input_dense = dim_input_dense
        self.dim_input_sparse = dim_input_sparse
        self.num_layers = num_layers
        self.tokenizer = SemanticTokenizer(group_dims, dim_emb, bias, dropout)
        num_tokens = len(group_dims) + 1
        self.blocks = [
            TokenMixerLargeBlock(
                dim_emb, num_heads, num_tokens, num_experts, top_k, bias=bias
            )
            for _ in range(num_layers)
        ]
        self.projection_head = MLP(
            dim_emb,
            num_hidden_head,
            dim_hidden_head,
            dim_output,
            dropout,
            bias,
        )
        self.aux_head = layers.Dense(dim_output, use_bias=bias)

    def call(self, inputs, training=False):
        sparse_inputs, dense_inputs = inputs
        x = self.embedding(sparse_inputs, dense_inputs)
        sparse_group = x[:, : self.dim_input_sparse, :]
        dense_group = x[
            :, self.dim_input_sparse : self.dim_input_sparse + self.dim_input_dense, :
        ]
        x = self.tokenizer([sparse_group, dense_group])
        is_last_layer = self.num_layers - 1
        residual_cache = []
        for i, layer in enumerate(self.blocks):
            x = layer(x)
            if i % 2 == 1 and i != is_last_layer:
                x = x + residual_cache[-1]
                residual_cache = []
            residual_cache.append(x)
            if training and i < is_last_layer and i % 2 == 1:
                aux_pooled = tf.reduce_mean(x, axis=1)
                aux_logit = self.aux_head(aux_pooled)
                self.add_loss(tf.reduce_mean(tf.square(aux_logit)))
        x = tf.reduce_mean(x, axis=1)
        x = self.projection_head(x)
        return x


if __name__ == "__main__":
    import numpy as np

    BATCH_SIZE = 2
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
    DIM_INPUT_SPARSE = 26
    DIM_INPUT_DENSE = 13

    # Example usage
    model = TokenMixerLarge(
        group_dims=[[128] * 26, [128] * 13],
        num_layers=2,
        num_sparse_embs=NUM_SPARSE_EMBS,
        dim_input_sparse=26,
        dim_input_dense=13,
        dim_emb=128,
        num_heads=4,
        num_experts=4,
        top_k=2,
        num_hidden_head=256,
        dim_hidden_head=128,
        dim_output=1,
    )

    sparse_inputs = tf.constant(
        np.column_stack(
            [
                np.random.randint(0, high=NUM_SPARSE_EMBS[i], size=BATCH_SIZE)
                for i in range(DIM_INPUT_SPARSE)
            ]
        ).astype(np.int32)
    )
    dense_inputs = tf.constant(
        np.random.rand(BATCH_SIZE, DIM_INPUT_DENSE).astype(np.float32)
    )

    outputs = model((sparse_inputs, dense_inputs))

    # 根据是否有辅助输出处理打印逻辑
    if isinstance(outputs, tuple):
        main_out, aux_outs = outputs
        print("Model main output shape:", main_out.shape)
        print(f"Number of auxiliary outputs: {len(aux_outs)}")
    else:
        print("Model output shape:", outputs.shape)
