# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import List, Tuple

from model.embedding import Embedding


class TowerMLP(layers.Layer):
    def __init__(
        self,
        hidden_units: Tuple[int, ...],
        activation: str,
        dropout: float,
        use_bias: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        tower_layers = []
        for units in hidden_units:
            tower_layers.append(layers.Dense(units, use_bias=use_bias))
            tower_layers.append(layers.Activation(activation))
            if dropout > 0.0:
                tower_layers.append(layers.Dropout(dropout))
        self.net = tf.keras.Sequential(tower_layers)
        self.logit = layers.Dense(1, use_bias=False)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.logit(self.net(x, training=training))


class ESMM(Model):
    """
    Entire Space Multi-Task Model.

    Outputs (p_ctr, p_cvr, p_ctcvr) — all probabilities in [0, 1].
    Training loss uses only p_ctr and p_ctcvr (Eq. 3).
    p_cvr is an intermediate variable and is NOT directly supervised.

    Args:
        num_sparse_embs:    Vocabulary sizes for each sparse feature.
        dim_emb:            Embedding dimension (paper: 18).
        dim_input_sparse:   Number of sparse features (26 for Criteo).
        dim_input_dense:    Number of dense features (13 for Criteo).
        tower_hidden_units: Hidden sizes for both CTR and CVR towers (paper: (360,200,80)).
        activation:         Tower activation (paper: relu).
        dropout:            Dropout rate.
        use_bias:           Whether Dense layers include bias.
    """

    def __init__(
        self,
        num_sparse_embs: List[int],
        dim_emb: int,
        dim_input_sparse: int,
        dim_input_dense: int,
        tower_hidden_units: Tuple[int, ...] = (360, 200, 80),
        activation: str = "relu",
        dropout: float = 0.0,
        use_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dim_input_sparse = dim_input_sparse
        self.dim_input_dense = dim_input_dense
        self.dim_emb = dim_emb
        self.flat_dim = (dim_input_sparse + dim_input_dense) * dim_emb
        self.output_names = ["p_ctr", "p_cvr", "p_ctcvr"]

        # Single shared embedding table for both CTR and CVR networks (Section 2.3)
        self.embedding = Embedding(num_sparse_embs, dim_emb, dim_input_dense, use_bias)

        # CTR tower — independent MLP weights
        self.ctr_tower = TowerMLP(
            tower_hidden_units, activation, dropout, use_bias, name="ctr_tower"
        )

        # CVR tower — independent MLP weights, embedding shared with CTR
        self.cvr_tower = TowerMLP(
            tower_hidden_units, activation, dropout, use_bias, name="cvr_tower"
        )

    def call(self, inputs, training: bool = False):
        """
        Args:
            inputs: (sparse_inputs [bs, S], dense_inputs [bs, D])
        Returns:
            p_ctr    (bs, 1)
            p_cvr    (bs, 1)  intermediate, not directly supervised
            p_ctcvr  (bs, 1)  = p_ctr * p_cvr
        """
        sparse_inputs, dense_inputs = inputs

        # Both towers consume the same embedding output (shared lookup table)
        embs = self.embedding(sparse_inputs, dense_inputs)  # (bs, S+D, dim_emb)
        flat = tf.reshape(embs, [-1, self.flat_dim])  # (bs, flat_dim)

        p_ctr = tf.sigmoid(self.ctr_tower(flat, training=training))  # (bs, 1)
        p_cvr = tf.sigmoid(self.cvr_tower(flat, training=training))  # (bs, 1)
        p_ctcvr = p_ctr * p_cvr  # (bs, 1)

        return p_ctr, p_cvr, p_ctcvr


if __name__ == "__main__":
    # Example usage for ESMM
    batch_size = 4
    num_sparse_embs = [100, 200, 150, 300, 50, 80]
    dim_emb = 18
    dim_input_sparse = len(num_sparse_embs)
    dim_input_dense = 5

    model = ESMM(
        num_sparse_embs=num_sparse_embs,
        dim_emb=dim_emb,
        dim_input_sparse=dim_input_sparse,
        dim_input_dense=dim_input_dense,
        tower_hidden_units=(360, 200, 80),
        activation="relu",
        dropout=0.0,
        use_bias=False,
    )

    sparse_inputs = tf.random.uniform(
        (batch_size, dim_input_sparse), minval=0, maxval=50, dtype=tf.int32
    )
    dense_inputs = tf.random.normal((batch_size, dim_input_dense))

    p_ctr, p_cvr, p_ctcvr = model((sparse_inputs, dense_inputs), training=False)
    print(f"ESMM p_ctr   shape: {p_ctr.shape}")
    print(f"ESMM p_cvr   shape: {p_cvr.shape}")
    print(f"ESMM p_ctcvr shape: {p_ctcvr.shape}")
