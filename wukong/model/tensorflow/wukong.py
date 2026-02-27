import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import List

from model.tensorflow.embedding import Embedding
from model.tensorflow.mlp import MLP


class LinearCompressBlock(layers.Layer):
    def __init__(
        self, num_emb_in: int, num_emb_out: int, bias: bool = False, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.num_emb_in = num_emb_in
        self.num_emb_out = num_emb_out
        self.linear = layers.Dense(num_emb_out, use_bias=bias)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # inputs: (bs, num_emb_in, dim_emb)
        x = tf.transpose(inputs, perm=[0, 2, 1])  # (bs, dim_emb, num_emb_in)
        x = self.linear(x)  # (bs, dim_emb, num_emb_out)
        x = tf.transpose(x, perm=[0, 2, 1])  # (bs, num_emb_out, dim_emb)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_emb_out, input_shape[2])


class FactorizationMachineBlock(layers.Layer):
    """TensorFlow implementation of Factorization Machine Block"""

    def __init__(
        self,
        num_emb_in: int,
        num_emb_out: int,
        dim_emb: int,
        rank: int,
        num_hidden: int,
        dim_hidden: int,
        dropout: float,
        bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_emb_in = num_emb_in
        self.num_emb_out = num_emb_out
        self.dim_emb = dim_emb
        self.rank = rank

        # Rank layer: (num_emb_in, rank)
        self.rank_layer = layers.Dense(
            rank, use_bias=bias, activation=None, name="rank_layer"
        )

        # Layer normalization
        self.norm = layers.LayerNormalization(name="layer_norm")

        # MLP for final transformation
        self.mlp = MLP(
            dim_in=num_emb_in * rank,
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=num_emb_out * dim_emb,
            dropout=dropout,
            bias=bias,
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass of the Factorization Machine Block

        Args:
            inputs: Tensor of shape (batch_size, num_emb_in, dim_emb)
            training: Boolean indicating training mode

        Returns:
            Tensor of shape (batch_size, num_emb_out, dim_emb)
        """
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        outputs = tf.transpose(inputs, perm=[0, 2, 1])

        # (bs, dim_emb, num_emb_in) @ (num_emb_in, rank) -> (bs, dim_emb, rank)
        outputs = self.rank_layer(outputs)

        # (bs, num_emb_in, dim_emb) @ (bs, dim_emb, rank) -> (bs, num_emb_in, rank)
        outputs = tf.matmul(inputs, outputs)

        # (bs, num_emb_in, rank) -> (bs, num_emb_in * rank)
        outputs = tf.reshape(outputs, [-1, self.num_emb_in * self.rank])

        # Layer normalization
        outputs = self.norm(outputs)

        # MLP transformation: (bs, num_emb_in * rank) -> (bs, num_emb_out * dim_emb)
        outputs = self.mlp(outputs, training=training)

        # (bs, num_emb_out * dim_emb) -> (bs, num_emb_out, dim_emb)
        outputs = tf.reshape(outputs, [-1, self.num_emb_out, self.dim_emb])

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_emb_in": self.num_emb_in,
                "num_emb_out": self.num_emb_out,
                "dim_emb": self.dim_emb,
                "rank": self.rank,
                "num_hidden": self.mlp.num_hidden,
                "dim_hidden": self.mlp.dim_hidden,
                "dropout": self.mlp.dropout_rate,
                "bias": self.mlp.use_bias,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class WukongLayer(layers.Layer):
    def __init__(
        self,
        num_emb_in: int,
        dim_emb: int,
        num_emb_lcb: int,
        num_emb_fmb: int,
        rank_fmb: int,
        num_hidden: int,
        dim_hidden: int,
        dropout: float,
        bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.lcb = LinearCompressBlock(num_emb_in, num_emb_lcb, bias)
        self.fmb = FactorizationMachineBlock(
            num_emb_in,
            num_emb_fmb,
            dim_emb,
            rank_fmb,
            num_hidden,
            dim_hidden,
            dropout,
            bias,
        )
        self.norm = layers.LayerNormalization(
            axis=-1
        )  # normalize over feature dimension

        if num_emb_in != num_emb_lcb + num_emb_fmb:
            self.residual_projection = LinearCompressBlock(
                num_emb_in, num_emb_lcb + num_emb_fmb, bias
            )
        else:
            self.residual_projection = layers.Lambda(lambda x: x)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_lcb, dim_emb)
        lcb = self.lcb(inputs)

        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_fmb, dim_emb)
        fmb = self.fmb(inputs)

        # (bs, num_emb_lcb, dim_emb), (bs, num_emb_fmb, dim_emb) -> (bs, num_emb_lcb + num_emb_fmb, dim_emb)
        outputs = tf.concat((fmb, lcb), axis=1)

        # (bs, num_emb_lcb + num_emb_fmb, dim_emb) -> (bs, num_emb_lcb + num_emb_fmb, dim_emb)
        residual = self.residual_projection(inputs)
        outputs = self.norm(outputs + residual)

        return outputs

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            self.lcb.num_emb_out + self.fmb.num_emb_out,
            input_shape[2],
        )


class Wukong(Model):
    def __init__(
        self,
        num_layers: int,
        num_sparse_embs: List[int],
        dim_emb: int,
        dim_input_sparse: int,
        dim_input_dense: int,
        num_emb_lcb: int,
        num_emb_fmb: int,
        rank_fmb: int,
        num_hidden_wukong: int,
        dim_hidden_wukong: int,
        num_hidden_head: int,
        dim_hidden_head: int,
        dim_output: int,
        dropout: float = 0.0,
        bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.dim_emb = dim_emb
        self.num_emb_lcb = num_emb_lcb
        self.num_emb_fmb = num_emb_fmb

        self.embedding = Embedding(num_sparse_embs, dim_emb, dim_input_dense, bias)

        num_emb_in = dim_input_sparse + dim_input_dense
        self.interaction_layers = []
        for _ in range(num_layers):
            layer = WukongLayer(
                num_emb_in,
                dim_emb,
                num_emb_lcb,
                num_emb_fmb,
                rank_fmb,
                num_hidden_wukong,
                dim_hidden_wukong,
                dropout,
                bias,
            )
            self.interaction_layers.append(layer)
            num_emb_in = num_emb_lcb + num_emb_fmb

        self.projection_head = MLP(
            (num_emb_lcb + num_emb_fmb) * dim_emb,
            num_hidden_head,
            dim_hidden_head,
            dim_output,
            dropout,
            bias,
        )

        # 将层添加到模型中，确保它们被正确跟踪
        self._layers = (
            [self.embedding] + self.interaction_layers + [self.projection_head]
        )
        # for exporting to ONNX
        self.output_names = ["output"]

    def call(self, inputs) -> tf.Tensor:
        sparse_inputs, dense_inputs = inputs
        outputs = self.embedding(sparse_inputs, dense_inputs)

        for layer in self.interaction_layers:
            outputs = layer(outputs)

        outputs = tf.reshape(
            outputs, [-1, (self.num_emb_lcb + self.num_emb_fmb) * self.dim_emb]
        )
        outputs = self.projection_head(outputs)

        return outputs

    def build(self, input_shape):
        sparse_shape, dense_shape = input_shape

        self.embedding.build([sparse_shape, dense_shape])

        dummy_sparse = tf.zeros(sparse_shape)
        dummy_dense = tf.zeros(dense_shape)
        emb_output = self.embedding(dummy_sparse, dummy_dense)

        current_output = emb_output
        for layer in self.interaction_layers:
            layer.build(current_output.shape)
            current_output = layer(current_output)

        final_shape = [None, (self.num_emb_lcb + self.num_emb_fmb) * self.dim_emb]
        self.projection_head.build(final_shape)
