from torch import nn


class MLP(nn.Sequential):
    def __init__(
        self,
        dim_in: int,
        num_hidden: int,
        dim_hidden: int,
        dim_out: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        layers = []
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(dim_in, dim_hidden, bias=bias))
            layers.append(nn.BatchNorm1d(dim_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            dim_in = dim_hidden

        layers.append(nn.Linear(dim_in, dim_out, bias=bias))

        super().__init__(*layers)
