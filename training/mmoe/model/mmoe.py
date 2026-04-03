# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import List, Optional, Tuple

from model.embedding import Embedding


class ExpertMLP(layers.Layer):
    def __init__(
        self,
        hidden_units: Tuple[int, ...],
        activation: str,
        dropout: float,
        use_bias: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        expert_layers = []
        for units in hidden_units:
            expert_layers.append(layers.Dense(units, use_bias=use_bias))
            expert_layers.append(layers.Activation(activation))
            if dropout > 0.0:
                expert_layers.append(layers.Dropout(dropout))
        self.net = tf.keras.Sequential(expert_layers)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.net(x, training=training)


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


class MMoE(Model):
    """
    Multi-gate Mixture-of-Experts multi-task model.

    Args:
        num_sparse_embs:    Vocabulary sizes for each sparse feature.
        dim_emb:            Embedding dimension.
        dim_input_sparse:   Number of sparse input features.
        dim_input_dense:    Number of dense input features.
        num_tasks:          Number of tasks.
        task_names:         Name for each task.
        num_experts:        Number of shared expert networks
        expert_hidden_units: Hidden sizes per expert MLP.
        tower_hidden_units:  Hidden sizes per tower MLP.
        activation:         Activation used in experts and towers.
        dropout:            Dropout rate.
        use_bias:           Whether Dense layers include bias.
    """

    def __init__(
        self,
        num_sparse_embs: List[int],
        dim_emb: int,
        dim_input_sparse: int,
        dim_input_dense: int,
        num_tasks: int = 2,
        task_names: Optional[List[str]] = None,
        num_experts: int = 8,
        expert_hidden_units: Tuple[int, ...] = (256, 128),
        tower_hidden_units: Tuple[int, ...] = (64,),
        activation: str = "relu",
        dropout: float = 0.0,
        use_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if task_names is None:
            task_names = [f"task{i}" for i in range(num_tasks)]
        assert len(task_names) == num_tasks

        self.num_tasks = num_tasks
        self.task_names = task_names
        self.dim_emb = dim_emb
        self.dim_input_sparse = dim_input_sparse
        self.dim_input_dense = dim_input_dense
        self.flat_dim = (dim_input_sparse + dim_input_dense) * dim_emb
        self.output_names = [f"output_{n}" for n in task_names]

        self.embedding = Embedding(num_sparse_embs, dim_emb, dim_input_dense, use_bias)

        # n shared expert networks (identical structure, independent parameters)
        self.experts: List[ExpertMLP] = [
            ExpertMLP(
                expert_hidden_units, activation, dropout, use_bias, name=f"expert_{i}"
            )
            for i in range(num_experts)
        ]

        # K gating networks: linear transformation + softmax
        self.gates: List[layers.Dense] = [
            layers.Dense(
                num_experts,
                use_bias=False,
                activation="softmax",
                name=f"gate_{task_names[k]}",
            )
            for k in range(num_tasks)
        ]

        # K task-specific tower networks
        self.towers: List[TowerMLP] = [
            TowerMLP(
                tower_hidden_units,
                activation,
                dropout,
                use_bias,
                name=f"tower_{task_names[k]}",
            )
            for k in range(num_tasks)
        ]

    def call(self, inputs, training: bool = False) -> List[tf.Tensor]:
        """
        Args:
            inputs: (sparse_inputs [bs, S], dense_inputs [bs, D])
        Returns:
            List of K tensors, each (bs, 1), raw logits.
        """
        sparse_inputs, dense_inputs = inputs

        flat = tf.reshape(
            self.embedding(sparse_inputs, dense_inputs),
            [-1, self.flat_dim],
        )  # (bs, flat_dim)

        # Run all experts once
        expert_outs = [e(flat, training=training) for e in self.experts]
        expert_stack = tf.stack(expert_outs, axis=1)  # (bs, n, expert_dim)

        logits = []
        for k in range(self.num_tasks):
            gate_w = self.gates[k](flat)  # (bs, n)
            gate_w = tf.expand_dims(gate_w, axis=-1)  # (bs, n, 1)
            fused = tf.reduce_sum(expert_stack * gate_w, axis=1)  # (bs, expert_dim)
            logits.append(self.towers[k](fused, training=training))

        return logits  # List of K tensors, each (bs, 1)


if __name__ == "__main__":
    # Example usage for MMoE
    batch_size = 4
    num_sparse_embs = [100, 200, 150, 300, 50, 80]
    dim_emb = 16
    dim_input_sparse = len(num_sparse_embs)
    dim_input_dense = 5

    model = MMoE(
        num_sparse_embs=num_sparse_embs,
        dim_emb=dim_emb,
        dim_input_sparse=dim_input_sparse,
        dim_input_dense=dim_input_dense,
        num_tasks=2,
        task_names=["ctr", "aux"],
        num_experts=8,
        expert_hidden_units=(256, 128),
        tower_hidden_units=(64,),
        activation="relu",
        dropout=0.0,
        use_bias=False,
    )

    sparse_inputs = tf.random.uniform(
        (batch_size, dim_input_sparse), minval=0, maxval=50, dtype=tf.int32
    )
    dense_inputs = tf.random.normal((batch_size, dim_input_dense))

    logits = model((sparse_inputs, dense_inputs), training=False)
    for name, logit in zip(model.task_names, logits):
        print(f"MMoE logits[{name}] shape: {logit.shape}")  # (4, 1)
