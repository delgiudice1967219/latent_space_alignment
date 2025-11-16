# src/models.py
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Type, Optional
from src.models.mlp_connector import MlpConnector


# Base Translator
class MLPTranslator(nn.Module):
    """
    Multi-layer perceptron that maps caption embeddings to image embeddings.

    :param in_dim: Dimension of the input text embedding.
    :param out_dim: Dimension of the target image embedding.
    :param hidden_dims: Width for each hidden layer.
    :param activation: Non-linearity used after each linear layer.
    :param use_residual: Whether to wrap equal-width layers in residual blocks.
    :param use_norm: Whether to add LayerNorm before linear layers.
    :param dropout: Dropout probability applied after activations.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        out_dim: int = 1536,
        hidden_dims: List[int] = [2048, 2048, 2048],
        activation: Type[nn.Module] = nn.GELU,
        use_residual: bool = True,
        use_norm: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        current_dim = in_dim

        # Input layer
        layers.append(nn.Linear(current_dim, hidden_dims[0]))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dims[0]

        # Hidden layers
        for h_dim in hidden_dims:
            block = self._create_block(
                current_dim, h_dim, activation, use_norm, dropout
            )
            if use_residual and current_dim == h_dim:
                layers.append(ResidualWrapper(block))
            else:
                layers.append(block)
            current_dim = h_dim

        # Output layer
        if use_norm:
            layers.append(nn.LayerNorm(current_dim))
        layers.append(nn.Linear(current_dim, out_dim))

        self.translator = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _create_block(self, input_dim, output_dim, activation, use_norm, dropout):
        layers = []
        if use_norm:
            layers.append(nn.LayerNorm(input_dim))
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Run the caption embeddings through the translator.

        :param x: Tensor of shape ``(batch, in_dim)``.
        :returns: Translated embeddings of shape ``(batch, out_dim)``.
        """
        return self.translator(x)


class ResidualWrapper(nn.Module):
    """
    Adds a skip connection around an arbitrary block.

    :param block: Module to wrap with ``x + block(x)``.
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x):
        """
        Apply the wrapped block and add the input back.

        :param x: Input tensor.
        :returns: Tensor with residual applied.
        """
        return x + self.block(x)


import torch
import torch.nn as nn


# Attention blocks
class CrossAttentionBlock(nn.Module):
    """
    Cross-attends absolute embeddings to relative embeddings.

    :param abs_dim: Dimension of the absolute embedding.
    :param rel_dim: Dimension of the relative embedding.
    :param attn_dim: Projection size used inside multi-head attention.
    :param num_heads: Number of attention heads.
    :param dropout: Dropout applied inside attention and projections.
    """

    def __init__(self, abs_dim, rel_dim, attn_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.query_proj = nn.Linear(abs_dim, attn_dim)
        self.key_proj = nn.Linear(rel_dim, attn_dim)
        self.value_proj = nn.Linear(rel_dim, attn_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.out_proj = nn.Linear(attn_dim, abs_dim)
        self.norm = nn.LayerNorm(abs_dim)

    def forward(self, x_abs, x_rel):
        """
        Attend absolute embeddings to one-step relative context.

        :param x_abs: Tensor ``(batch, abs_dim)``.
        :param x_rel: Tensor ``(batch, rel_dim)``.
        :returns: Updated absolute embeddings of shape ``(batch, abs_dim)``.
        """
        q = self.query_proj(x_abs).unsqueeze(1)  # [B, 1, attn_dim]
        k = self.key_proj(x_rel).unsqueeze(1)  # treat rel features as single "token"
        v = self.value_proj(x_rel).unsqueeze(1)
        attn_out, _ = self.attn(q, k, v)  # [B, 1, attn_dim]
        attn_out = self.out_proj(attn_out.squeeze(1))
        return self.norm(x_abs + attn_out)


class BatchSelfAttention(nn.Module):
    """
    Self-attention applied across the batch to add contextual signal.

    :param dim: Input feature dimension.
    :param attn_dim: Internal attention dimension (defaults to ``dim``).
    :param num_heads: Number of attention heads.
    :param dropout: Dropout applied within attention.
    """

    def __init__(self, dim, attn_dim=None, num_heads=4, dropout=0.1):
        super().__init__()
        attn_dim = attn_dim or dim
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.q_proj = nn.Linear(dim, attn_dim)
        self.k_proj = nn.Linear(dim, attn_dim)
        self.v_proj = nn.Linear(dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Apply batch-level self-attention.

        :param x: Tensor ``(batch, dim)``; each sample is treated as a token.
        :returns: Tensor of shape ``(batch, dim)`` with contextual update.
        """
        q = self.q_proj(x).unsqueeze(0)
        k = self.k_proj(x).unsqueeze(0)
        v = self.v_proj(x).unsqueeze(0)
        attn_out, _ = self.attn(q, k, v)
        attn_out = self.out_proj(attn_out.squeeze(0))
        return self.norm(x + attn_out)


# Translator model v2
class TranslatorModelV2(nn.Module):
    """
    Translator with optional cross-attention and batch self-attention.

    :param abs_dim: Dimension of absolute text embeddings.
    :param rel_dim: Dimension of relative text embeddings.
    :param output_dim: Dimension of the predicted image embeddings.
    :param hidden_dims: Hidden layer sizes for the MLP encoder.
    :param dropout: Dropout probability for encoder blocks.
    :param use_cross_attn: Enable cross-attention between abs/rel parts.
    :param use_batch_attn: Enable self-attention across the batch.
    :param cross_attn_heads: Number of heads for cross-attention.
    :param cross_attn_dim: Projection size inside cross-attention.
    :param batch_attn_heads: Number of heads for batch self-attention.
    :param batch_attn_dim: Projection size inside batch attention.
    """

    def __init__(
        self,
        abs_dim=1024,
        rel_dim=256,
        output_dim=1536,
        hidden_dims=(2048, 2048, 2048),
        dropout=0.2,
        use_cross_attn=False,
        use_batch_attn=False,
        cross_attn_heads=4,
        cross_attn_dim=512,
        batch_attn_heads=4,
        batch_attn_dim=512,
    ):
        super().__init__()

        self.use_cross_attn = use_cross_attn
        self.use_batch_attn = use_batch_attn

        # Optional cross-attention block
        if use_cross_attn:
            self.cross_attn = CrossAttentionBlock(
                abs_dim=abs_dim,
                rel_dim=rel_dim,
                attn_dim=cross_attn_dim,
                num_heads=cross_attn_heads,
                dropout=dropout,
            )
        else:
            self.cross_attn = None

        # Input dimension for MLP (concat if no cross-attn)
        in_dim = abs_dim + (0 if use_cross_attn else rel_dim)

        # Build residual MLP encoder from hidden_dims list
        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h
        self.encoder = nn.Sequential(*layers)

        # Optional batch attention
        if use_batch_attn:
            self.batch_attn = BatchSelfAttention(
                dim=hidden_dims[-1],
                attn_dim=batch_attn_dim,
                num_heads=batch_attn_heads,
                dropout=dropout,
            )
        else:
            self.batch_attn = None

        # Projection to output embedding space
        self.head = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a batch through the model.

        The input is expected to concatenate ``[absolute | relative]`` features
        when cross-attention is enabled.

        :param x: Tensor ``(batch, abs_dim + rel_dim)``.
        :returns: Predicted embeddings of shape ``(batch, output_dim)``.
        """
        if self.use_cross_attn:
            # split at 1024 (absolute dim)
            abs_dim = getattr(self, "abs_dim", 1024)
            x_abs, x_rel = x[:, :abs_dim], x[:, abs_dim:]
            x = self.cross_attn(x_abs, x_rel)
        elif self.use_batch_attn:
            # if only batch attention enabled, keep x as-is
            pass
        else:
            # standard MLP (concat or absolute-only)
            pass

        # 2. Residual MLP encoder
        x = self.encoder(x)

        # 3. Optional batch-level attention
        if self.batch_attn:
            x = self.batch_attn(x)

        # 4. Output projection
        return self.head(x)


# DeepTranslator
class ResidualFFN(nn.Module):
    """
    Pre-normalized feed-forward block with optional residual skip.

    :param dim_in: Input feature dimension.
    :param dim_out: Output feature dimension.
    :param activation: Activation class to use.
    :param use_norm: Whether to apply LayerNorm before the linear layer.
    :param dropout: Dropout probability after activation.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        activation: Type[nn.Module] = nn.GELU,
        use_norm: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_norm = use_norm
        self.norm = nn.LayerNorm(dim_in) if use_norm else nn.Identity()
        self.linear = nn.Linear(dim_in, dim_out)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)
        self.use_residual = dim_in == dim_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the feed-forward block.

        :param x: Tensor of shape ``(batch, dim_in)``.
        :returns: Tensor of shape ``(batch, dim_out)`` with residual if matched.
        """
        y = x
        # y = self.norm(x)
        y = self.linear(y)
        y = self.activation(y)
        y = self.dropout(y)
        return x + y if self.use_residual else y


class DeepTranslator(nn.Module):
    """
    Deeper MLP variant with residual FFN blocks and global skip.

    :param in_dim: Input text embedding dimension.
    :param out_dim: Output image embedding dimension.
    :param hidden_dims: Hidden sizes for stacked FFN blocks.
    :param activation: Activation class to use in blocks.
    :param use_norm: Whether to apply LayerNorm before the head.
    :param dropout: Dropout probability inside blocks.
    :param normalize: L2-normalize the output embeddings when ``True``.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        out_dim: int = 1536,
        hidden_dims: List[int] = [2048, 2048, 2048, 2048],
        activation: Type[nn.Module] = nn.GELU,
        use_norm: bool = True,
        dropout: float = 0.2,
        normalize: bool = True,
    ):
        super().__init__()
        self.normalize = normalize

        layers = []
        current_dim = in_dim

        # Input projection
        layers.append(nn.Linear(in_dim, hidden_dims[0]))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dims[0]

        # Residual FFN blocks
        for h_dim in hidden_dims:
            layers.append(
                ResidualFFN(
                    dim_in=current_dim,
                    dim_out=h_dim,
                    activation=activation,
                    use_norm=use_norm,
                    dropout=dropout,
                )
            )
            current_dim = h_dim

        # Output projection
        if use_norm:
            layers.append(nn.LayerNorm(current_dim))
        layers.append(nn.Linear(current_dim, out_dim))

        self.network = nn.Sequential(*layers)
        self.shortcut = nn.Linear(in_dim, out_dim, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Translate captions to image-space embeddings.

        :param x: Tensor ``(batch, in_dim)``.
        :returns: Tensor ``(batch, out_dim)``; normalized if ``normalize`` is set.
        """
        y = self.network(x)
        y = y + self.shortcut(x)
        if self.normalize:
            y = F.normalize(y, p=2, dim=-1)
        return y


# AdaLN1DBlock
class AdaLN1DBlock(nn.Module):
    """
    1D AdaLN residual block with optional cross-sample self-attention.

    :param in_dim: Input feature dimension.
    :param hidden_dim: Hidden size used inside the MLP.
    :param cond_dim: Conditioning dimension for FiLM modulation.
    :param out_dim: Output feature dimension (defaults to ``in_dim``).
    :param dropout: Dropout applied inside the block.
    :param use_self_attn: Enable batch-level self-attention.
    :param attn_heads: Number of heads for the optional attention.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        cond_dim: int,
        out_dim: int | None = None,
        dropout: float = 0.0,
        use_self_attn: bool = False,
        attn_heads: int = 8,
    ):
        super().__init__()
        out_dim = out_dim or in_dim

        # Main MLP path
        self.norm = nn.LayerNorm(in_dim)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        # FiLM-style conditioning
        self.cond = nn.Sequential(
            nn.Linear(cond_dim, max(out_dim // 2, 1)),
            nn.SiLU(),
            nn.Linear(max(out_dim // 2, 1), out_dim * 2),
        )

        # Skip projection (if needed)
        self.proj = None
        if in_dim != out_dim:
            self.proj = nn.Linear(in_dim, out_dim)

        # Initialize last layer for stability
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

        # Optional self-attention
        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.attn_norm = nn.LayerNorm(out_dim)
            self.self_attn = nn.MultiheadAttention(
                embed_dim=out_dim,
                num_heads=attn_heads,
                batch_first=True,  # (B, seq_len, D)
            )
            self.attn_dropout = nn.Dropout(dropout)
            # small residual scaling to keep training stable
            self.attn_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Apply AdaLN modulation and optional self-attention.

        :param x: Input tensor ``(batch, in_dim)``.
        :param cond: Conditioning tensor ``(batch, cond_dim)``.
        :returns: Updated tensor ``(batch, out_dim)``.
        """
        #  AdaLN MLP path ---
        h = self.norm(x)
        h = self.fc(h)
        gamma, beta = self.cond(cond).chunk(2, dim=-1)
        h = (1 + gamma) * h + beta
        skip = x if self.proj is None else self.proj(x)
        h = h + skip

        # Optional batch-level self-attention ---
        if self.use_self_attn:
            # Treat the batch as a "sequence" of B tokens
            h_norm = self.attn_norm(h).unsqueeze(1)  # (B, 1, D)
            attn_out, _ = self.self_attn(h_norm, h_norm, h_norm)
            attn_out = attn_out.squeeze(1)
            h = h + self.attn_scale * self.attn_dropout(attn_out)

        return h


# AdaLNTranslator
class AdaLNTranslator(nn.Module):
    """
    Translator that stacks AdaLN blocks and can finish with a Transformer tail.

    :param text_dim: Total dimension of the input text embedding.
    :param abs_text_dim: Dimension of the absolute portion of the embedding.
    :param width: Hidden width used throughout the network.
    :param depth: Number of AdaLN blocks.
    :param out_dim: Dimension of the predicted image embedding.
    :param dropout: Dropout probability for MLPs and attention.
    :param normalize_out: L2-normalize outputs when ``True``.
    :param add_linear_skip: Add a linear skip from absolute embeddings.
    :param use_transformer_tail: Enable Transformer refinement across batch tokens.
    :param n_transformer_layers: Number of layers in the Transformer tail.
    :param attn_heads: Number of attention heads in the Transformer tail.
    """

    def __init__(
        self,
        text_dim: int = 1024,  # total input dim (may include anchors)
        abs_text_dim: int = 1024,  # absolute text embedding dim
        width: int = 768,
        depth: int = 4,
        out_dim: int = 1536,
        dropout: float = 0.1,
        normalize_out: bool = True,
        add_linear_skip: bool = True,
        use_transformer_tail: bool = False,  # <── new flag
        n_transformer_layers: int = 2,  # <── tail depth
        attn_heads: int = 8,  # <── heads for transformer tail
    ) -> None:
        super().__init__()

        self.normalize_out = normalize_out
        self.add_linear_skip = add_linear_skip
        self.abs_text_dim = abs_text_dim
        self.use_transformer_tail = use_transformer_tail

        # --- Input projection ---
        self.in_proj = nn.Linear(text_dim, width)

        # --- Core AdaLN residual stack ---
        self.blocks = nn.ModuleList(
            [
                AdaLN1DBlock(
                    in_dim=width,
                    hidden_dim=2 * width,
                    cond_dim=abs_text_dim,
                    out_dim=width,
                    dropout=dropout,
                    use_self_attn=False,  # now disabled inside blocks
                )
                for _ in range(depth)
            ]
        )

        # --- Optional Transformer tail ---
        if use_transformer_tail:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=attn_heads,
                dim_feedforward=2 * width,
                dropout=dropout,
                batch_first=True,  # (B, seq_len, D)
                activation="gelu",
                norm_first=True,
            )
            self.transformer_tail = nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_transformer_layers,
            )
            self.tail_norm = nn.LayerNorm(width)

        # --- Output head ---
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, out_dim),
        )

        # --- Optional linear skip from absolute embedding ---
        self.linear_skip = None
        if add_linear_skip:
            self.linear_skip = nn.Linear(abs_text_dim, out_dim, bias=False)
            nn.init.orthogonal_(self.linear_skip.weight)

    def forward(self, z_text: torch.Tensor) -> torch.Tensor:
        """
        Translate text embeddings into image embeddings.

        :param z_text: Tensor ``(batch, text_dim)`` or ``(text_dim,)``.
        :returns: Tensor ``(batch, out_dim)`` in image space.
        """
        if z_text.dim() == 1:
            z_text = z_text.unsqueeze(0)

        # Split absolute and relative parts
        z_abs = z_text[:, : self.abs_text_dim]

        # Encode with AdaLN blocks
        h = self.in_proj(z_text)
        for block in self.blocks:
            h = block(h, z_abs)

        # Optional Transformer tail (global refinement)
        if self.use_transformer_tail:
            # treat batch as "sequence" (B tokens)
            h_seq = h.unsqueeze(1)  # (B, 1, D)
            h_refined = self.transformer_tail(h_seq).squeeze(1)  # (B, D)
            h = h + self.tail_norm(h_refined)  # residual refinement

        # Output projection
        y = self.head(h)

        # Linear skip from abs embedding
        if self.linear_skip is not None:
            y = y + self.linear_skip(z_abs)

        if self.normalize_out:
            y = F.normalize(y, dim=-1)

        return y
