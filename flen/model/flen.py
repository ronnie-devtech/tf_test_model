import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import List, Tuple

from model.embedding import Embedding


class DiceFactor(layers.Layer):
    """
    Bernoulli path-dropout for FM bi-linear interactions.
    Train : each element of the input vector kept with probability beta.
    Infer : multiply whole vector by beta (mean-network scheme).
    """

    def __init__(self, beta: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        if training:
            mask = tf.cast(tf.random.uniform(tf.shape(x)) < self.beta, dtype=tf.float32)
            return x * mask
        return x * self.beta

    def get_config(self):
        cfg = super().get_config()
        cfg["beta"] = self.beta
        return cfg


class FieldWiseBiInteraction(layers.Layer):
    """
    Computes hMF + hFM through a ReLU hidden layer → hFwBI (bs, dim_emb).

    The S sub-module (linear scalar hS) is computed outside this layer and
    added to the final logit, following the reference implementation pattern.

    Args:
        num_fields : M, number of hierarchical field groups.
        dim_emb    : Ke, embedding dimension.
        beta       : DiceFactor keep probability.
        use_bias   : Whether the hidden Dense uses bias.

    Input:
        List of M tensors, each (bs, n_m, Ke) — per-field feature embeddings.
    Output:
        (bs, Ke) — hFwBI after ReLU hidden layer.
    """

    def __init__(
        self,
        num_fields: int,
        dim_emb: int,
        beta: float = 0.7,
        use_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_fields = num_fields
        self.dim_emb = dim_emb

        self.dicefactor = DiceFactor(beta, name="dicefactor")

        # Hidden layer: maps (hMF + hFM) ∈ R^Ke → hFwBI ∈ R^Ke
        self.hidden = layers.Dense(
            dim_emb, use_bias=use_bias, activation="relu", name="fwbi_hidden"
        )

    def build(self, input_shape):
        M = self.num_fields
        n_inter = M * (M - 1) // 2

        self.r_inter = self.add_weight(
            name="r_inter",
            shape=(n_inter,),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.r_intra = self.add_weight(
            name="r_intra",
            shape=(M,),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Trigger Dense weight creation with a dummy call path
        dummy = tf.zeros((1, self.dim_emb))
        self.hidden(dummy)

        super().build(input_shape)

    def call(
        self,
        inputs: List[tf.Tensor],  # M x (bs, n_m, Ke)
        training: bool = False,
    ) -> tf.Tensor:  # (bs, Ke)
        # Field-level sum pooling: em = sum_{F(n)=m} en
        field_sums = [tf.reduce_sum(inp, axis=1) for inp in inputs]
        # field_sums: M x (bs, Ke)

        h_mf = tf.zeros_like(field_sums[0])  # (bs, Ke)
        idx = 0
        for i in range(self.num_fields):
            for j in range(i + 1, self.num_fields):
                h_mf = h_mf + field_sums[i] * field_sums[j] * self.r_inter[idx]
                idx += 1

        h_fm = tf.zeros_like(field_sums[0])  # (bs, Ke)
        for m in range(self.num_fields):
            hfm = field_sums[m] * field_sums[m]  # (bs, Ke)
            htm = tf.reduce_sum(inputs[m] * inputs[m], axis=1)  # (bs, Ke)
            diff = self.dicefactor(hfm - htm, training=training)  # DiceFactor
            h_fm = h_fm + diff * self.r_intra[m]

        return self.hidden(h_mf + h_fm)  # (bs, Ke)


class FLEN(Model):
    """
    Field-Leveraged Embedding Network.

    Args:
        num_sparse_embs  : Vocabulary sizes for each sparse feature.
        dim_emb          : Embedding dimension Ke (paper: 32).
        dim_input_sparse : Number of sparse features (26 for Criteo).
        dim_input_dense  : Number of dense features (13 for Criteo).
        field_groups     : List of M lists; each sub-list contains the
                           embedding-tensor column indices belonging to
                           that hierarchical field group.
                           Indices range over [0, dim_input_sparse+dim_input_dense).
        dnn_hidden_units : MLP hidden layer sizes .
        dicefactor_beta  : DiceFactor keep probability).
        dropout          : MLP dropout rate.
        use_bias         : Whether Dense layers include bias.
    """

    def __init__(
        self,
        num_sparse_embs: List[int],
        dim_emb: int,
        dim_input_sparse: int,
        dim_input_dense: int,
        field_groups: List[List[int]],
        dnn_hidden_units: Tuple[int, ...] = (64, 32),
        dicefactor_beta: float = 0.7,
        dropout: float = 0.0,
        use_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dim_emb = dim_emb
        self.dim_input_sparse = dim_input_sparse
        self.dim_input_dense = dim_input_dense
        self.field_groups = [list(g) for g in field_groups]
        self.M = len(field_groups)
        self.output_names = ["output"]

        # Embedding layer
        self.embedding = Embedding(num_sparse_embs, dim_emb, dim_input_dense, use_bias)

        # FwBI pooling layer
        self.fwbi = FieldWiseBiInteraction(
            num_fields=self.M,
            dim_emb=dim_emb,
            beta=dicefactor_beta,
            use_bias=use_bias,
            name="fwbi",
        )

        # input: concat of M field sum embeddings → (bs, M*Ke)
        mlp_layers: List[layers.Layer] = []
        for units in dnn_hidden_units:
            mlp_layers.append(layers.Dense(units, use_bias=use_bias))
            mlp_layers.append(layers.Activation("relu"))
            if dropout > 0.0:
                mlp_layers.append(layers.Dropout(dropout))
        self.mlp = tf.keras.Sequential(mlp_layers, name="mlp")

        # S sub-module: linear logit Dense(1) on field sum concat
        # Approximates hS = w0 + sum wi[j]*xi[j] using dense embeddings
        self.linear = layers.Dense(1, use_bias=True, name="linear_s")

        # z = w^T_F * concat(hFwBI, hMLP)
        self.pred_layer = layers.Dense(1, use_bias=False, name="pred_layer")

    def call(self, inputs, training: bool = False) -> tf.Tensor:
        """
        Args:
            inputs: (sparse_inputs [bs, S], dense_inputs [bs, D])
        Returns:
            logit (bs, 1) — raw logit; apply sigmoid for probability.
        """
        sparse_inputs, dense_inputs = inputs

        # Embedding: (bs, S+D, Ke)
        all_embs = self.embedding(sparse_inputs, dense_inputs)

        # Split embedding tensor into M field groups: each (bs, n_m, Ke)
        field_emb_list = [tf.gather(all_embs, grp, axis=1) for grp in self.field_groups]

        # Field-level sum embeddings for MLP and linear inputs
        field_sums = [tf.reduce_sum(e, axis=1) for e in field_emb_list]
        # M x (bs, Ke)

        # S sub-module: linear scalar logit
        mlp_input = tf.concat(field_sums, axis=-1)  # (bs, M*Ke)
        linear_logit = self.linear(mlp_input)  # (bs, 1)

        # FwBI: hFwBI (bs, Ke)
        h_fwbi = self.fwbi(field_emb_list, training=training)

        # MLP component: hMLP (bs, last_hidden_dim)
        h_mlp = self.mlp(mlp_input, training=training)

        # Prediction: linear_logit (hS) + Dense(1)(concat(hFwBI, hMLP))
        dnn_logit = self.pred_layer(tf.concat([h_fwbi, h_mlp], axis=-1))  # (bs, 1)

        return linear_logit + dnn_logit  # (bs, 1)


if __name__ == "__main__":
    # Example usage for FLEN
    batch_size = 4
    num_sparse_embs = [100, 200, 150, 300, 50, 80, 40, 60, 90]
    dim_emb = 32
    dim_input_sparse = len(num_sparse_embs)  # 9
    dim_input_dense = 4  # total emb indices: 0-12
    field_groups = [
        list(range(0, 3)),  # group 0
        list(range(3, 6)),  # group 1
        list(range(6, dim_input_sparse + dim_input_dense)),  # group 2
    ]

    model = FLEN(
        num_sparse_embs=num_sparse_embs,
        dim_emb=dim_emb,
        dim_input_sparse=dim_input_sparse,
        dim_input_dense=dim_input_dense,
        field_groups=field_groups,
        dnn_hidden_units=(64, 32),
        dicefactor_beta=0.7,
        dropout=0.0,
        use_bias=False,
    )

    sparse_inputs = tf.random.uniform(
        (batch_size, dim_input_sparse), minval=0, maxval=50, dtype=tf.int32
    )
    dense_inputs = tf.random.normal((batch_size, dim_input_dense))

    logit_train = model((sparse_inputs, dense_inputs), training=True)
    logit_infer = model((sparse_inputs, dense_inputs), training=False)

    print(f"FLEN logit shape (train): {logit_train.shape}")  # (4, 1)
    print(f"FLEN logit shape (infer): {logit_infer.shape}")  # (4, 1)

    # DiceFactor: train outputs should differ from infer (stochastic vs scaled)
    prob_train = tf.sigmoid(logit_train).numpy()
    prob_infer = tf.sigmoid(logit_infer).numpy()
    print(f"p_train[:2]: {prob_train[:2, 0]}")
    print(f"p_infer[:2]: {prob_infer[:2, 0]}")
