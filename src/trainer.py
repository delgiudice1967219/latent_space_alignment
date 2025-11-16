# src/trainer.py
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import List
from src.losses import LossFactory, compute_similarity


class TranslationTrainer(pl.LightningModule):
    """
    Lightning module that trains and evaluates a text-to-image translator.

    Validation loaders follow the Lightning multi-dataloader convention:
    dataloader ``0`` logs loss on all captions, while dataloaders ``1..V``
    stream caption views aligned to the validation gallery for MRR reporting.

    Logged metrics are grouped to mirror leaderboard expectations (per-view,
    per-fold, and overall aggregates).
    """

    def __init__(self, model, config, val_data):
        super().__init__()
        self.model = model
        self.config = config

        # Gallery for MRR (aligned and truncated)
        self.register_buffer("val_image_gallery_mrr", val_data["gallery"])

        # Standardization stats (optional)
        mu_y = val_data.get("mu_y", None)
        std_y = val_data.get("std_y", None)
        if mu_y is not None and std_y is not None:
            # Ensure they are tensors and match device placement
            self.register_buffer("mu_y", mu_y.clone().detach())
            self.register_buffer("std_y", std_y.clone().detach())
        else:
            self.mu_y, self.std_y = None, None

        # Containers for MRR aggregation
        self.mrr_chunks_all_views: List[List[float]] = []
        self.current_view_chunks: List[float] = []
        self.mrr_mixed_equal_chunks: List[float] = (
            []
        )  # for the new mixed MRR dataloader

        # Validation settings
        self.val_mrr_batch_size = int(config.get("val_mrr_batch_size", 100))
        self.val_mrr_folds = int(config.get("val_mrr_folds", 8))
        self.val_num_caption_views = int(config.get("val_num_caption_views", 3))
        self.log_chunks = bool(config.get("val_log_chunks", False))

        self.loss_name = config.get("loss_name", "infonce").lower()
        self.similarity_type = config.get("similarity_type", "dot")
        self.symmetric = bool(config.get("symmetric", False))
        self.multi_positive = bool(config.get("multi_positive", False))
        self.use_mse = bool(config.get("use_mse", False))
        self._lambda_mse = float(config.get("lambda_mse", 3.7))

        # Optional kernel parameters
        self.alpha = float(config.get("alpha", 1.0))
        self.degree = int(config.get("degree", 2))

        # Optional schedulers
        self.alpha_scheduler = config.get("alpha_scheduler", None)
        self.batch_scheduler = config.get("batch_scheduler", None)

        # Optional LR scheduler
        self.use_lr_scheduler = bool(config.get("use_lr_scheduler", False))
        self.lr_scheduler_type = config.get("lr_scheduler_type", "cosine")
        self.save_hyperparameters(ignore=["model", "val_data"])

        loss_params = {
            k: v
            for k, v in config.items()
            if k not in ["optimizer", "scheduler", "trainer"]
        }
        self.loss_fn = LossFactory.build(self.loss_name, **loss_params)

        self.compute_similarity = compute_similarity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Translate input embeddings with the wrapped model.

        :param x: Caption embeddings of shape ``(batch, dim)``.
        :returns: Predicted image embeddings of shape ``(batch, 1536)``.
        """
        return self.model(x)

    def on_train_epoch_start(self):
        """
        Update schedulers that depend on the current epoch and refresh
        the datamodule batch composition if a batch scheduler is provided.
        """
        # Update alpha scheduler
        if self.alpha_scheduler is not None:
            self.alpha = self.alpha_scheduler(self.current_epoch)

        # Update batch composition
        if self.batch_scheduler is not None and hasattr(self.trainer, "datamodule"):
            mode = self.batch_scheduler.get_mode(self.current_epoch)
            self.trainer.datamodule.update_batch_mode(**mode)

    def compute_loss(self, y_pred, y_true, labels=None):
        """
        Compute the training/validation loss for a batch.

        :param y_pred: Predicted embeddings (captions mapped to image space).
        :param y_true: Ground-truth image embeddings.
        :param labels: Optional image ids to enable multi-positive handling.
        :returns: Scalar loss value.
        """
        # Similarity matrix (caption to image)
        sims = self.compute_similarity(
            y_pred,
            y_true,
            type=self.similarity_type,
            alpha=self.alpha,
            degree=self.degree,
        )

        # Forward loss (caption to image)
        loss_fwd = self.loss_fn(sims, labels=labels if self.multi_positive else None)

        # Symmetric loss (image to caption) using transposed similarities
        if self.symmetric:
            loss_back = self.loss_fn(
                sims.T, labels=labels if self.multi_positive else None
            )
            loss = 0.5 * (loss_fwd + loss_back)
        else:
            loss = loss_fwd

        if self.use_mse:
            mse_loss = F.mse_loss(
                F.normalize(y_pred, dim=1), F.normalize(y_true, dim=1)
            )
            loss = loss + (self._lambda_mse * mse_loss)

        return loss

    def training_step(self, batch, batch_idx):
        """
        Perform one training step on a batch of caption-image pairs.

        :param batch: Tuple ``(captions, images, labels)``.
        :param batch_idx: Batch index within the epoch.
        :returns: Training loss tensor.
        """
        x, y_true, labels = batch

        y_pred = self.model(x)
        loss = self.compute_loss(y_pred, y_true, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # Validation
    def validation_step(self, batch, batch_idx, dataloader_idx):
        """
        Run validation loss or MRR, depending on which dataloader is active.

        :param batch: Batch from the active validation dataloader.
        :param batch_idx: Batch index within that dataloader.
        :param dataloader_idx: Index identifying whether this is loss (0) or MRR (>=1).
        """
        # Validation loss
        if dataloader_idx == 0:
            x, y_true, labels = batch
            y_pred = self.model(x)
            loss = self.compute_loss(y_pred, y_true, labels)
            self.log("val/loss", loss, prog_bar=True)
            return loss

        # MRR evaluations (views + mixed-equal)
        (x_captions,) = batch
        B = x_captions.size(0)
        start = batch_idx * B
        pred = self.model(x_captions)
        local_gallery = self.val_image_gallery_mrr[start : start + B]

        # Optional de-standardization
        if self.mu_y is not None and self.std_y is not None:
            pred = pred * self.std_y + self.mu_y
            local_gallery = local_gallery * self.std_y + self.mu_y

        # Cosine-sim/dot-product matrix
        sims = pred @ local_gallery.T
        topk_local = torch.topk(sims, k=B, dim=1, largest=True, sorted=True).indices
        gt_global = start + torch.arange(B, device=self.device)
        pred_global = (start + topk_local).cpu().numpy()
        gt_global_np = gt_global.cpu().numpy()

        # Manual MRR computation
        rr = []
        for i in range(B):
            hits = np.where(pred_global[i] == gt_global_np[i])[0]
            rr.append(0.0 if hits.size == 0 else 1.0 / (hits[0] + 1))
        mrr_chunk = float(np.mean(rr))

        # Standard per-view MRR
        if 1 <= dataloader_idx <= self.val_num_caption_views:
            view_id = dataloader_idx - 1
            if len(self.mrr_chunks_all_views) <= view_id:
                self.mrr_chunks_all_views.append([])
            self.mrr_chunks_all_views[view_id].append(mrr_chunk)

            if self.log_chunks:
                self.log(f"chunks/view_{view_id + 1}_chunk_{batch_idx + 1}", mrr_chunk)

        # Mixed-equal MRR (added dataloader)
        elif dataloader_idx == (1 + self.val_num_caption_views):
            self.mrr_mixed_equal_chunks.append(mrr_chunk)

    def on_validation_epoch_end(self):
        """
        Aggregate chunk-level Mean Reciprocal Ranks into folds, views,
        and overall averages for logging.
        """

        # Per-view aggregation
        if self.mrr_chunks_all_views:
            view_means, view_stds = [], []

            for v, chunks in enumerate(self.mrr_chunks_all_views, start=1):
                chunks = np.asarray(chunks, dtype=np.float32)

                # Group chunks into folds
                num_chunks = len(chunks)
                if self.val_mrr_folds > 0:
                    fold_size = num_chunks // self.val_mrr_folds or 1
                    folds = [
                        chunks[i * fold_size : (i + 1) * fold_size].mean()
                        for i in range(self.val_mrr_folds)
                        if i * fold_size < num_chunks
                    ]
                else:
                    folds = [chunks.mean()]

                folds = np.asarray(folds)
                mrr_view_avg = float(folds.mean())
                mrr_view_std = float(folds.std(ddof=0))
                view_means.append(mrr_view_avg)
                view_stds.append(mrr_view_std)

                # Logging
                self.log(f"views/view_{v}_avg", mrr_view_avg)
                self.log(f"views/view_{v}_std", mrr_view_std)

                if self.config.get("log_folds", False):
                    for f_idx, mrr_fold in enumerate(folds, start=1):
                        self.log(f"folds/view_{v}_fold_{f_idx}", float(mrr_fold))

            # Overall aggregates (in "val/")
            view_means = np.asarray(view_means)
            overall_avg = float(view_means.mean())
            overall_std_views = float(view_means.std(ddof=0))
            overall_std_folds = float(np.mean(view_stds))

            self.log("val/mrr_overall_avg", overall_avg, prog_bar=True)
            self.log("val/mrr_overall_std_views", overall_std_views)
            self.log("val/mrr_overall_std_folds", overall_std_folds)

            # Reset for next epoch
            self.mrr_chunks_all_views.clear()

        # Mixed-equal MRR aggregation (new)
        if hasattr(self, "mrr_mixed_equal_chunks") and self.mrr_mixed_equal_chunks:
            chunks = np.asarray(self.mrr_mixed_equal_chunks, dtype=np.float32)

            # Aggregate across all mixed batches
            mrr_mixed_avg = float(chunks.mean())
            mrr_mixed_std = float(chunks.std(ddof=0))

            # Log under the main "val/" section
            self.log("val/mrr_mixed_equal_avg", mrr_mixed_avg)
            self.log("val/mrr_mixed_equal_std", mrr_mixed_std)

            # Reset buffer
            self.mrr_mixed_equal_chunks.clear()

    # Prediction / optimizer setup
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Forward pass for inference, optionally de-standardizing outputs.

        :param batch: Tuple containing caption embeddings.
        :param batch_idx: Batch index.
        :param dataloader_idx: Dataloader identifier (unused, kept for Lightning API).
        :returns: Predicted image embeddings.
        """
        x = batch[0]
        y_pred = self.model(x)

        if self.mu_y is not None and self.std_y is not None:
            y_pred = y_pred * self.std_y + self.mu_y

        return y_pred

    def configure_optimizers(self):
        """
        Configure AdamW optimizer and an optional learning-rate scheduler.

        :returns: Optimizer or dict with scheduler configuration for Lightning.
        :raises ValueError: If an unknown scheduler type is requested.
        """
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["optimizer"]["lr"],
            weight_decay=self.config["optimizer"]["weight_decay"],
        )

        if not self.use_lr_scheduler:
            return opt

        if self.lr_scheduler_type == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=self.trainer.max_epochs,
                eta_min=self.config["optimizer"]["lr"] * 0.01,
            )
        elif self.lr_scheduler_type == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=0.5, patience=5, verbose=True
            )
        else:
            raise ValueError(f"Unknown lr_scheduler_type: {self.lr_scheduler_type}")

        return {"optimizer": opt, "lr_scheduler": sched}
