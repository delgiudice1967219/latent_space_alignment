# src/losses.py
import torch
import torch.nn.functional as F
from typing import Optional


# Utility: similarity computation
def compute_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
    type: str = "dot",
    alpha: float = 1.0,
    degree: int = 2,
):
    """
    Compute a pairwise similarity/kernel matrix between two embedding sets.

    :param a: Tensor of shape ``(N, D)``.
    :param b: Tensor of shape ``(N, D)``.
    :param type: Similarity to use: ``"dot"``, ``"cosine"``, ``"poly_kernel"``, or ``"rbf_kernel"``.
    :param alpha: Kernel scale (used for polynomial and RBF kernels).
    :param degree: Polynomial degree when ``type="poly_kernel"``.
    :returns: Similarity matrix of shape ``(N, N)``.
    :raises ValueError: If an unknown ``type`` is provided.
    """
    if type == "dot":
        sims = a @ b.T
    elif type == "cosine":
        a = F.normalize(a, p=2, dim=1)
        b = F.normalize(b, p=2, dim=1)
        sims = a @ b.T
    elif type == "poly_kernel":
        sims = (alpha * (a @ b.T) + 1.0).pow(degree)
    elif type == "rbf_kernel":
        a2 = (a**2).sum(dim=1, keepdim=True)
        b2 = (b**2).sum(dim=1, keepdim=True)
        dist2 = a2 + b2.T - 2 * (a @ b.T)
        sims = torch.exp(-alpha * dist2)
    else:
        raise ValueError(f"Unknown similarity type: {type}")
    return sims


# Base class for all losses
class BaseLoss:
    """Interface for similarity-based losses."""

    def __call__(
        self, sims: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError


# InfoNCE Loss (supports multi-positive)
class InfoNCELoss(BaseLoss):
    def __init__(self, temperature: float = 0.07):
        self.temperature = temperature

    def __call__(
        self, sims: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE with optional multi-positive handling.

        :param sims: Similarity matrix of shape ``(B, B)``.
        :param labels: Optional labels to treat matching rows/cols as positives.
        :returns: Scalar loss.
        """
        logits = sims / self.temperature
        B = sims.size(0)

        if labels is None:
            # Standard single-positive InfoNCE
            targets = torch.arange(B, device=sims.device)
            return F.cross_entropy(logits, targets)

        # Multi-positive: build label mask
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask.fill_diagonal_(False)

        log_prob = F.log_softmax(logits, dim=1)
        loss = -(log_prob * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return loss.mean()


# Triplet Loss (cosine-based, batch hard/semi/soft)
class TripletLoss(BaseLoss):
    def __init__(self, margin: float = 0.2, mining: str = "hard"):  # , top_k: int = 1
        self.margin = margin
        self.mining = mining
        # self.top_k = top_k

    def __call__(
        self, sims: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cosine-based triplet loss with configurable mining.

        :param sims: Similarity matrix where the diagonal represents positives.
        :param labels: Optional labels to mask out false negatives.
        :returns: Scalar loss.
        :raises ValueError: If the mining strategy is unknown.
        """
        B = sims.size(0)
        pos_sim = sims.diag()

        # Mask false negatives
        if labels is not None:
            mask = torch.ones_like(sims, dtype=torch.bool)
            same = labels.unsqueeze(0) == labels.unsqueeze(1)
            mask[same] = False
            mask.fill_diagonal_(False)
            neg_sims = sims.masked_fill(~mask, float("-inf"))
        else:
            neg_sims = sims.masked_fill(
                torch.eye(B, dtype=torch.bool, device=sims.device), float("-inf")
            )

        # Mining strategy
        if self.mining == "hard":
            hardest_neg = neg_sims.max(dim=1).values
            loss = F.relu(self.margin + hardest_neg - pos_sim).mean()

        elif self.mining == "semi":
            med = neg_sims.median(dim=1, keepdim=True).values
            semi_mask = neg_sims >= med
            semi_mean = (neg_sims * semi_mask).sum(dim=1) / semi_mask.sum(
                dim=1
            ).clamp_min(1)
            loss = F.relu(self.margin + semi_mean - pos_sim).mean()

        elif self.mining == "soft":
            valid = torch.isfinite(neg_sims)
            loss = F.relu(
                self.margin
                + (
                    neg_sims.masked_fill(~valid, 0).sum(dim=1)
                    / valid.sum(dim=1).clamp_min(1)
                )
                - pos_sim
            ).mean()

        else:
            raise ValueError(f"Unknown mining strategy: {self.mining}")

        return loss


# Circle Loss
class CircleLoss(BaseLoss):
    def __init__(self, m: float = 0.25, gamma: float = 80.0):
        self.m = m
        self.gamma = gamma

    def __call__(
        self, sims: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Circle loss that balances positive and negative pairs.

        :param sims: Similarity matrix where the diagonal represents positives.
        :param labels: Optional labels to ignore false negatives.
        :returns: Scalar loss combining positive and negative penalties.
        """
        N = sims.size(0)
        pos = sims.diag()

        # Build valid negative mask
        if labels is not None:
            mask = torch.ones_like(sims, dtype=torch.bool)
            same = labels.unsqueeze(0) == labels.unsqueeze(1)
            mask[same] = False
            mask.fill_diagonal_(False)
            neg = sims.masked_fill(~mask, float("-inf"))
        else:
            neg = sims.masked_fill(
                torch.eye(N, dtype=torch.bool, device=sims.device), float("-inf")
            )

        # Circle Loss formulation
        op, on = 1 + self.m, -self.m
        alpha_p = torch.clamp_min(op - pos.detach(), 0.0)
        alpha_n = torch.clamp_min(neg.detach() - on, 0.0)

        logit_p = -self.gamma * alpha_p * (pos - (1 - self.m))
        logit_n = self.gamma * alpha_n * (neg - (-self.m))

        valid_neg = torch.isfinite(logit_n)
        logit_n = logit_n.masked_fill(~valid_neg, 0.0)

        loss_p = torch.log1p(torch.exp(logit_p)).mean()
        loss_n = torch.log1p(torch.exp(logit_n[valid_neg])).mean()
        return loss_p + loss_n


# Factory
class LossFactory:
    @staticmethod
    def build(name: str, **kwargs):
        """
        Build a loss instance by name.

        :param name: One of ``\"infonce\"``, ``\"triplet\"``, or ``\"circle\"``.
        :param kwargs: Extra parameters forwarded to the loss constructor.
        :returns: Instantiated loss object.
        :raises ValueError: If ``name`` is not recognized.
        """
        name = name.lower()
        if name in ["infonce", "nce", "contrastive"]:
            return InfoNCELoss(
                **{k: v for k, v in kwargs.items() if k in ["temperature"]}
            )
        elif name == "triplet":
            return TripletLoss(
                **{k: v for k, v in kwargs.items() if k in ["margin", "mining"]}
            )
        elif name == "circle":
            return CircleLoss(
                **{k: v for k, v in kwargs.items() if k in ["m", "gamma"]}
            )
        else:
            raise ValueError(f"Unknown loss name: {name}")
