# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import List, Tuple, Optional

from model.embedding import Embedding


# ──────────────────────────────────────────────────────────────────────────────
#  Expert Network
# ──────────────────────────────────────────────────────────────────────────────


class ExpertMLP(layers.Layer):
    """
    Single expert sub-network.
    """

    def __init__(
        self,
        hidden_units: Tuple[int, ...] = (256,),
        activation: str = "relu",
        dropout: float = 0.0,
        use_bias: bool = False,
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

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.net(inputs, training=training)


# ──────────────────────────────────────────────────────────────────────────────
#  CGC Extraction Layer  (one level of PLE)
# ──────────────────────────────────────────────────────────────────────────────


class CGCLayer(layers.Layer):
    """
    Customized Gate Control (CGC) extraction layer — one level of PLE.

    For each task k:
        S^k(x)  = concat of task-k specific experts + shared experts
        w^k(x)  = Softmax(W^k_g · x)           — single-layer gate
        g^k(x)  = w^k(x) · S^k(x)              — weighted sum pooling

    For the shared module (non-last levels only):
        S^s(x)  = ALL task-specific experts + shared experts
        shared gate analogous to task gate
    """

    def __init__(
        self,
        num_tasks: int,
        shared_expert_num: int,
        specific_expert_num: int,
        expert_hidden_units: Tuple[int, ...],
        activation: str,
        dropout: float,
        use_bias: bool,
        is_last: bool,
        level_idx: int,
        task_names: List[str],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_tasks = num_tasks
        self.shared_expert_num = shared_expert_num
        self.specific_expert_num = specific_expert_num
        self.is_last = is_last

        # ── Task-specific experts ─────────────────────────────────────────
        # self.specific_experts[i][j] = j-th expert for task i
        self.specific_experts: List[List[ExpertMLP]] = []
        for i in range(num_tasks):
            row = [
                ExpertMLP(
                    expert_hidden_units,
                    activation,
                    dropout,
                    use_bias,
                    name=f"level{level_idx}_{task_names[i]}_specific_expert{j}",
                )
                for j in range(specific_expert_num)
            ]
            self.specific_experts.append(row)

        # ── Shared experts ────────────────────────────────────────────────
        self.shared_experts: List[ExpertMLP] = [
            ExpertMLP(
                expert_hidden_units,
                activation,
                dropout,
                use_bias,
                name=f"level{level_idx}_shared_expert{k}",
            )
            for k in range(shared_expert_num)
        ]

        # ── Task gating networks: Dense(m_k + m_s) + Softmax ─────────────
        # Each task gate selects from its own specific experts + shared experts
        n_task_gate = specific_expert_num + shared_expert_num
        self.task_gates: List[layers.Dense] = [
            layers.Dense(
                n_task_gate,
                use_bias=False,
                activation="softmax",
                name=f"level{level_idx}_gate_{task_names[i]}",
            )
            for i in range(num_tasks)
        ]

        # ── Shared gating network (only for non-last levels) ──────────────
        # Shared gate selects from ALL specific experts + shared experts
        if not is_last:
            n_shared_gate = num_tasks * specific_expert_num + shared_expert_num
            self.shared_gate: Optional[layers.Dense] = layers.Dense(
                n_shared_gate,
                use_bias=False,
                activation="softmax",
                name=f"level{level_idx}_gate_shared",
            )
        else:
            self.shared_gate = None

    def call(
        self,
        inputs: List[tf.Tensor],  # [task_0, ..., task_{K-1}, shared]
        training: bool = False,
    ) -> List[tf.Tensor]:
        """
        Returns:
            is_last=True  → [task_0_out, ..., task_{K-1}_out]
            is_last=False → [task_0_out, ..., task_{K-1}_out, shared_out]
        """
        # ── Run all experts ───────────────────────────────────────────────
        # spec_outs[i][j] : (bs, expert_dim)
        spec_outs: List[List[tf.Tensor]] = []
        for i in range(self.num_tasks):
            row = [
                self.specific_experts[i][j](inputs[i], training=training)
                for j in range(self.specific_expert_num)
            ]
            spec_outs.append(row)

        # shared_outs[k] : (bs, expert_dim)
        shared_outs: List[tf.Tensor] = [
            self.shared_experts[k](inputs[-1], training=training)
            for k in range(self.shared_expert_num)
        ]

        cgc_outputs: List[tf.Tensor] = []

        # ── Task-specific gating ────────────────────────────────
        for i in range(self.num_tasks):
            # Gather: task-specific experts for task i + all shared experts
            cur_experts = spec_outs[i] + shared_outs  # list of (bs, E)
            expert_stack = tf.stack(cur_experts, axis=1)  # (bs, mk+ms, E)

            gate_w = self.task_gates[i](inputs[i])  # (bs, mk+ms)
            gate_w = tf.expand_dims(gate_w, axis=-1)  # (bs, mk+ms, 1)

            out = tf.reduce_sum(expert_stack * gate_w, axis=1)  # (bs, E)
            cgc_outputs.append(out)

        # ── Shared gating ──────────────────────────
        if not self.is_last:
            all_spec_flat: List[tf.Tensor] = []
            for i in range(self.num_tasks):
                all_spec_flat.extend(spec_outs[i])
            all_experts = all_spec_flat + shared_outs
            expert_stack = tf.stack(all_experts, axis=1)  # (bs, all_n, E)

            gate_w = self.shared_gate(inputs[-1])  # (bs, all_n)
            gate_w = tf.expand_dims(gate_w, axis=-1)

            out = tf.reduce_sum(expert_stack * gate_w, axis=1)  # (bs, E)
            cgc_outputs.append(out)

        return cgc_outputs


# ──────────────────────────────────────────────────────────────────────────────
#  Tower Network
# ──────────────────────────────────────────────────────────────────────────────


class TowerNetwork(layers.Layer):
    """
    Task-specific tower: MLP → linear logit.

    Paper Section 5.1.3: three-layer MLP [256, 128, 64] + ReLU per task.
    Final Dense(1, no bias) outputs a raw logit (BCE from_logits=True).
    """

    def __init__(
        self,
        hidden_units: Tuple[int, ...] = (256, 128, 64),
        activation: str = "relu",
        dropout: float = 0.0,
        use_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        tower_layers = []
        for units in hidden_units:
            tower_layers.append(layers.Dense(units, use_bias=use_bias))
            tower_layers.append(layers.Activation(activation))
            if dropout > 0.0:
                tower_layers.append(layers.Dropout(dropout))
        self.tower = tf.keras.Sequential(tower_layers)
        self.logit = layers.Dense(1, use_bias=False)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.tower(inputs, training=training)
        return self.logit(x)  # (bs, 1)


# ──────────────────────────────────────────────────────────────────────────────
#  PLE Model
# ──────────────────────────────────────────────────────────────────────────────


class PLE(Model):
    """
    Progressive Layered Extraction (PLE) multi-task model.

    Args:
        num_sparse_embs:      List of vocabulary sizes per sparse feature.
        dim_emb:              Embedding dimension for all features.
        dim_input_sparse:     Number of sparse features (26 for Criteo).
        dim_input_dense:      Number of dense features (13 for Criteo).
        num_tasks:            Number of tasks (2 for the Criteo adaptation).
        task_names:           Names for each task (for weight naming only).
        shared_expert_num:    Number of shared expert networks per CGC level.
        specific_expert_num:  Number of task-specific experts per task per level.
        num_levels:           Number of CGC extraction levels.
        expert_hidden_units:  Hidden unit sizes for each expert MLP.
        tower_hidden_units:   Hidden unit sizes for each tower MLP.
        expert_activation:    Activation used inside expert networks.
        tower_activation:     Activation used inside tower networks.
        dropout:              Dropout rate applied in experts and towers.
        use_bias:             Whether Dense layers use bias terms.
    """

    def __init__(
        self,
        num_sparse_embs: List[int],
        dim_emb: int,
        dim_input_sparse: int,
        dim_input_dense: int,
        num_tasks: int = 2,
        task_names: Optional[List[str]] = None,
        shared_expert_num: int = 1,
        specific_expert_num: int = 1,
        num_levels: int = 2,
        expert_hidden_units: Tuple[int, ...] = (256,),
        tower_hidden_units: Tuple[int, ...] = (256, 128, 64),
        expert_activation: str = "relu",
        tower_activation: str = "relu",
        dropout: float = 0.0,
        use_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if task_names is None:
            task_names = [f"task{i}" for i in range(num_tasks)]
        assert (
            len(task_names) == num_tasks
        ), f"len(task_names)={len(task_names)} != num_tasks={num_tasks}"

        self.num_tasks = num_tasks
        self.task_names = task_names
        self.dim_emb = dim_emb
        self.dim_input_sparse = dim_input_sparse
        self.dim_input_dense = dim_input_dense
        self.flat_dim = (dim_input_sparse + dim_input_dense) * dim_emb
        self.output_names = [f"output_{n}" for n in task_names]

        self.embedding = Embedding(num_sparse_embs, dim_emb, dim_input_dense, use_bias)

        # ── N CGC extraction layers ────────────────────────────────────────
        self.cgc_layers: List[CGCLayer] = []
        for level in range(num_levels):
            is_last = level == num_levels - 1
            self.cgc_layers.append(
                CGCLayer(
                    num_tasks=num_tasks,
                    shared_expert_num=shared_expert_num,
                    specific_expert_num=specific_expert_num,
                    expert_hidden_units=expert_hidden_units,
                    activation=expert_activation,
                    dropout=dropout,
                    use_bias=use_bias,
                    is_last=is_last,
                    level_idx=level,
                    task_names=task_names,
                    name=f"cgc_level{level}",
                )
            )

        # ── Task-specific tower networks ───────────────────────────────────
        self.towers: List[TowerNetwork] = [
            TowerNetwork(
                tower_hidden_units,
                tower_activation,
                dropout,
                use_bias,
                name=f"tower_{task_names[i]}",
            )
            for i in range(num_tasks)
        ]

    # ── Forward pass ──────────────────────────────────────────────────────

    def call(self, inputs, training: bool = False) -> List[tf.Tensor]:
        """
        Args:
            inputs: (sparse_inputs, dense_inputs)
                sparse_inputs : (bs, dim_input_sparse)  int32
                dense_inputs  : (bs, dim_input_dense)   float32
        Returns:
            List of K tensors, each (bs, 1) — raw logits (before sigmoid).
        """
        sparse_inputs, dense_inputs = inputs

        # ── Embedding → flatten ────────────────────────────────────────────
        all_embs = self.embedding(sparse_inputs, dense_inputs)  # (bs, 39, D)
        flat = tf.reshape(all_embs, [-1, self.flat_dim])  # (bs, 39*D)

        # ── PLE: all tasks + shared start from the same flat representation ─
        # ple_inputs = [task_0_repr, ..., task_{K-1}_repr, shared_repr]
        ple_inputs: List[tf.Tensor] = [flat] * (self.num_tasks + 1)

        # ── Pass through N CGC levels ──────────────────────────────────────
        for cgc in self.cgc_layers:
            ple_inputs = cgc(ple_inputs, training=training)

        # ── Tower networks → logits ────────────────────────────────────────
        logits = [
            self.towers[i](ple_inputs[i], training=training)  # (bs, 1)
            for i in range(self.num_tasks)
        ]

        return logits  # list of K tensors, each (bs, 1)


# ──────────────────────────────────────────────────────────────────────────────
#  Example usage
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example usage for PLE
    batch_size = 4
    num_sparse_embs = [100, 200, 150, 300, 50, 80]
    dim_emb = 16
    dim_input_sparse = len(num_sparse_embs)  # 6
    dim_input_dense = 5

    model = PLE(
        num_sparse_embs=num_sparse_embs,
        dim_emb=dim_emb,
        dim_input_sparse=dim_input_sparse,
        dim_input_dense=dim_input_dense,
        num_tasks=2,
        task_names=["ctr", "aux"],
        shared_expert_num=1,
        specific_expert_num=1,
        num_levels=2,
        expert_hidden_units=(256,),
        tower_hidden_units=(256, 128, 64),
        expert_activation="relu",
        tower_activation="relu",
        dropout=0.0,
        use_bias=False,
    )

    # Dummy input data
    sparse_inputs = tf.random.uniform(
        (batch_size, dim_input_sparse), minval=0, maxval=50, dtype=tf.int32
    )
    dense_inputs = tf.random.normal((batch_size, dim_input_dense))

    logits = model((sparse_inputs, dense_inputs), training=False)
    for i, (name, logit) in enumerate(zip(model.task_names, logits)):
        print(f"PLE logits[{name}] shape: {logit.shape}")  # (4, 1)
