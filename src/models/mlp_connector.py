# src/models/mlp_conncetor.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpConnector(nn.Module):
    """
    Two-layer MLP that maps text embeddings into the image embedding space.

    :param input_dim: Dimensionality of the incoming text embedding.
    :param hidden_dim: Width of the hidden layer.
    :param output_dim: Dimensionality of the projected embedding.
    :param dropout_rate: Dropout probability applied after the activation.
    :param normalize_out: If ``True``, L2-normalize the output vectors.
    :param layer_norm: If ``True``, include a layer norm after the first linear.
    :param activation_fun: Activation class to instantiate between the two linears.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 2048,
        output_dim: int = 1536,
        dropout_rate: float = 0.1,
        normalize_out: bool = True,
        layer_norm: bool = True,
        activation_fun=nn.SiLU,
    ):
        super().__init__()
        self.normalize_out = normalize_out
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else None,
            activation_fun(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )
        print(f"Created MLPConnector: {input_dim} -> {hidden_dim} -> {output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the connector on a batch of embeddings.

        :param x: Input tensor of shape ``(batch, input_dim)``.
        :returns: Projected embeddings of shape ``(batch, output_dim)``; normalized if configured.
        """
        y = self.net(x)
        if self.normalize_out:
            y = F.normalize(y, dim=-1)
        return y
