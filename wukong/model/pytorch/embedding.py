import torch
from torch import Tensor, nn
from typing import List


class SparseEmbedding(nn.Module):
    def __init__(self, num_sparse_embs: List[int], dim_emb: int) -> None:
        super().__init__()

        self.embeddings = nn.ModuleList()
        for num_emb in num_sparse_embs:
            self.embeddings.append(nn.Embedding(num_emb, dim_emb))

    def forward(self, sparse_inputs: Tensor) -> Tensor:
        sparse_outputs = []
        for i, embedding in enumerate(self.embeddings):
            sparse_outputs.append(embedding(sparse_inputs[:, i]))
        return torch.stack(sparse_outputs, dim=1)


class Embedding(nn.Module):
    def __init__(
        self,
        num_sparse_embs: List[int],
        dim_emb: int,
        dim_input_dense: int,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.dim_emb = dim_emb
        self.dim_input_dense = dim_input_dense

        self.sparse_embedding = SparseEmbedding(num_sparse_embs, dim_emb)
        self.dense_embedding = nn.Linear(
            dim_input_dense, dim_input_dense * dim_emb, bias=bias
        )

    def forward(self, sparse_inputs: Tensor, dense_inputs) -> Tensor:
        sparse_outputs = self.sparse_embedding(sparse_inputs)
        dense_outputs = self.dense_embedding(dense_inputs).view(
            -1, self.dim_input_dense, self.dim_emb
        )

        return torch.concat((sparse_outputs, dense_outputs), dim=1)
