# src/data.py
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Sampler
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional, Iterator
from collections import defaultdict


class GroupedImageBatchSampler(Sampler):
    """
    Batch sampler that groups captions by image id.

    :param train_image_ids_sorted: Sorted array of unique training image ids.
    :param img_id_to_caption_indices: Map from image id to the 5 caption indices.
    :param images_per_batch: Number of images (i.e., groups of 5 captions) per batch.
    :param shuffle: If ``True``, shuffle image order each epoch.
    """

    def __init__(
        self,
        train_image_ids_sorted: np.ndarray,
        img_id_to_caption_indices: Dict[int, List[int]],
        images_per_batch: int,
        shuffle: bool = True,
    ) -> None:

        self.num_images = len(train_image_ids_sorted)
        self.img_id_to_caption_indices = img_id_to_caption_indices
        self.images_per_batch = images_per_batch
        self.shuffle = shuffle

        self.index_to_original_img_id = {
            idx: img_id for idx, img_id in enumerate(train_image_ids_sorted)
        }

        if self.num_images < self.images_per_batch:
            raise ValueError(
                f"Not enough images ({self.num_images}) to form a single batch of {self.images_per_batch} images."
            )

    def __iter__(self) -> Iterator[List[int]]:
        """
        Yield the caption indices that compose each batch.

        :returns: Iterator of lists, each of length ``images_per_batch * 5``.
        """
        image_indices = np.arange(self.num_images)

        if self.shuffle:
            np.random.shuffle(image_indices)

        # Iterate over the image indices in chunks
        for i in range(0, len(image_indices), self.images_per_batch):
            batch_image_indices = image_indices[i : i + self.images_per_batch]

            # Stop if we have an incomplete batch
            if len(batch_image_indices) < self.images_per_batch:
                break

            batch_caption_indices: List[int] = []
            for img_idx in batch_image_indices:
                # Get the original image id from the 0-based index
                original_img_id = self.index_to_original_img_id[img_idx]

                # Look up the caption indices (row numbers) for that image
                caption_indices_for_img = self.img_id_to_caption_indices[
                    original_img_id
                ]

                # Add all 5 caption indices to the batch list
                batch_caption_indices.extend(caption_indices_for_img)

            yield batch_caption_indices

    def __len__(self) -> int:
        """
        Number of full batches per epoch (drops the remainder).

        :returns: Integer count of batches.
        """
        # Drop the last incomplete batch
        return self.num_images // self.images_per_batch


class HardImageBatchSampler(Sampler):
    """
    Batch sampler that builds batches of semantically similar images.

    :param faiss_index: Pre-built FAISS index over training image embeddings.
    :param train_images_unique: Unique training image embeddings aligned with ``faiss_index``.
    :param train_image_ids_sorted: Sorted array of unique training image ids.
    :param img_id_to_caption_indices: Map from image id to the 5 caption indices.
    :param images_per_batch: Number of images (groups of captions) per batch.
    :param over_sample_factor: Multiplier controlling how many neighbors to query from FAISS.
    :raises ValueError: If the FAISS index size does not match the images or if a batch cannot be formed.
    """

    def __init__(
        self,
        faiss_index,
        train_images_unique: torch.Tensor,
        train_image_ids_sorted: np.ndarray,
        img_id_to_caption_indices: Dict[int, List[int]],
        images_per_batch: int,
        over_sample_factor: int = 5,
    ) -> None:

        self.num_images = len(train_image_ids_sorted)
        if self.num_images != faiss_index.ntotal:
            raise ValueError("FAISS index size does not match number of unique images.")

        self.faiss_index = faiss_index
        self.train_images_unique = train_images_unique
        self.img_id_to_caption_indices = img_id_to_caption_indices
        self.images_per_batch = images_per_batch

        self.over_sample_factor = int(over_sample_factor)
        if self.over_sample_factor < 2:
            print(
                f"Warning: HardImageBatchSampler over_sample_factor={self.over_sample_factor} is small. "
                "Batches may fill with random images late in the epoch."
            )

        # Calculate the actual k to query from FAISS
        k_to_query = self.images_per_batch * self.over_sample_factor
        self.k_to_query = min(k_to_query, self.num_images)

        if self.k_to_query < self.images_per_batch:
            print(
                f"Warning: k_to_query ({self.k_to_query}) is less than images_per_batch "
                f"({self.images_per_batch}). Setting to {min(self.images_per_batch, self.num_images)}."
            )
            self.k_to_query = min(self.images_per_batch, self.num_images)

        self.index_to_original_img_id = {
            idx: img_id for idx, img_id in enumerate(train_image_ids_sorted)
        }

        if self.num_images < self.images_per_batch:
            raise ValueError(
                f"Not enough images ({self.num_images}) to form a single batch of {self.images_per_batch} images."
            )

    def __iter__(self) -> Iterator[List[int]]:
        """
        Yield caption indices grouped via hard-negative mining.

        :returns: Iterator of lists, each with ``images_per_batch * 5`` indices.
        """
        image_indices_shuffled = np.random.permutation(self.num_images)

        # Track which images have been used in a batch
        visited_mask = np.zeros(self.num_images, dtype=bool)

        # Use a pointer to walk through the shuffled list
        idx_ptr = 0
        while idx_ptr < self.num_images:

            # Find the next unvisited image to be our "anchor"
            anchor_index = image_indices_shuffled[idx_ptr]
            idx_ptr += 1
            if visited_mask[anchor_index]:
                continue

            # Start the batch with this anchor
            current_batch_indices = [anchor_index]
            visited_mask[anchor_index] = True

            # Query FAISS for its nearest neighbors
            anchor_emb = (
                self.train_images_unique[anchor_index].cpu().numpy().reshape(1, -1)
            )
            anchor_emb /= np.linalg.norm(anchor_emb, axis=1, keepdims=True) + 1e-8
            D, I = self.faiss_index.search(anchor_emb, k=self.k_to_query)
            D, I = self.faiss_index.search(anchor_emb, k=self.k_to_query)

            # Add unvisited neighbors to the batch (from the sorted list)
            for neighbor_index in I[0]:
                if len(current_batch_indices) >= self.images_per_batch:
                    break
                # Skip self-match (if neighbor_index is anchor_index)
                if neighbor_index == anchor_index:
                    continue
                if not visited_mask[neighbor_index]:
                    current_batch_indices.append(neighbor_index)
                    visited_mask[neighbor_index] = True

            # If batch is still not full (not enough unvisited neighbors),
            # fill it with random unvisited images from our shuffled list
            while (
                len(current_batch_indices) < self.images_per_batch
                and idx_ptr < self.num_images
            ):
                fill_index = image_indices_shuffled[idx_ptr]
                idx_ptr += 1
                if not visited_mask[fill_index]:
                    current_batch_indices.append(fill_index)
                    visited_mask[fill_index] = True

            # If the batch is still not full, drop this incomplete batch.
            if len(current_batch_indices) < self.images_per_batch:
                continue

            # We have a full batch of image indices. Convert to caption indices.
            batch_caption_indices: List[int] = []
            for img_idx in current_batch_indices:
                original_img_id = self.index_to_original_img_id[img_idx]
                caption_indices_for_img = self.img_id_to_caption_indices[
                    original_img_id
                ]
                batch_caption_indices.extend(caption_indices_for_img)

            yield batch_caption_indices

    def __len__(self) -> int:
        """
        Number of full batches per epoch (drops the remainder).

        :returns: Integer count of batches.
        """
        # We drop the last incomplete batch
        return self.num_images // self.images_per_batch


class EmbeddingDataModule(pl.LightningDataModule):
    """
    Data module for translating text embeddings to image embeddings.

    :param data_path: Folder containing the task npz files.
    :param coco_npz_path: Optional COCO npz to concatenate before splitting.
    :param batch_size: Training batch size (must be a multiple of 5 when grouping captions).
    :param seed: Seed used for deterministic splits and view selection.
    :param standardize: If ``True``, standardize embeddings using training stats.
    :param val_mrr_batch_size: Batch size for validation MRR (Kaggle uses 100).
    :param val_mrr_folds: Number of 100-query chunks (folds) for MRR.
    :param val_num_caption_views: Number of caption views (one caption per image per view).
    :param num_anchors: Number of anchor embeddings concatenated to captions.
    :param group_captions: If ``True``, batch 5 captions per image together.
    :param sample_hard_images: If ``True``, use FAISS to build hard-image batches.
    """

    def __init__(
        self,
        data_path: str,
        coco_npz_path: Optional[str] = None,
        batch_size: int = 128,
        seed: int = 42,
        standardize: bool = False,
        *,
        val_mrr_batch_size: int = 100,
        val_mrr_folds: int = 15,
        val_num_caption_views: int = 5,
        num_anchors: int = 0,
        group_captions: bool = False,
        sample_hard_images: bool = False,
    ) -> None:

        super().__init__()
        self._setup_complete = False

        self.group_captions = bool(group_captions)
        self.sample_hard_images = bool(sample_hard_images)

        if self.sample_hard_images and not self.group_captions:
            raise ValueError(
                "'sample_hard_images=True' requires 'group_captions=True', "
                "as hard sampling is an image-level strategy."
            )

        if self.group_captions:
            if batch_size % 5 != 0:
                raise ValueError(
                    "When 'group_captions=True', batch_size must be a multiple of 5."
                )
            self.images_per_batch = batch_size // 5

        if val_num_caption_views > 5:
            raise ValueError(
                f"val_num_caption_views={val_num_caption_views} exceeds maximum (5 captions per image)."
            )

        self.val_mrr_batch_size = int(val_mrr_batch_size)
        self.val_mrr_folds = int(val_mrr_folds)
        self.val_size = self.val_mrr_batch_size * self.val_mrr_folds  # inferred
        self.val_num_caption_views = int(val_num_caption_views)
        self.num_anchors = int(num_anchors)

        self.data_path = data_path
        self.coco_npz_path = coco_npz_path
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.standardize = bool(standardize)

        self.val_image_gallery: Optional[torch.Tensor] = None
        self.val_image_gallery_mrr: Optional[torch.Tensor] = None
        self.X_val_mrr_views: List[torch.Tensor] = []

        self.X_train: Optional[torch.Tensor] = None
        self.y_train: Optional[torch.Tensor] = None
        self.labels_train: Optional[torch.Tensor] = None

        self.X_val: Optional[torch.Tensor] = None
        self.y_val: Optional[torch.Tensor] = None
        self.labels_val: Optional[torch.Tensor] = None

        self.X_test: Optional[torch.Tensor] = None
        self.test_ids: Optional[np.ndarray] = None

        self.mu_x = self.std_x = self.mu_y = self.std_y = None

        self.train_images_unique: Optional[torch.Tensor] = None
        self.train_image_ids_sorted: Optional[np.ndarray] = None
        self.train_img_id_to_caption_indices: Optional[Dict[int, List[int]]] = None
        self.faiss_index = None  # Will be imported if needed

    # Helpers
    def _build_image_structures(
        self, Y_unique, train_image_ids, label_ids_all, is_train_caption
    ):
        """
        Build lookup tables for grouping captions by image.

        :param Y_unique: Array of unique image embeddings.
        :param train_image_ids: Image ids assigned to training.
        :param label_ids_all: Image ids for every caption.
        :param is_train_caption: Boolean mask selecting training captions.
        """
        print("Building image-centric structures for training sampler...")
        from collections import defaultdict

        self.train_image_ids_sorted = np.sort(train_image_ids)
        self.train_images_unique = torch.from_numpy(
            Y_unique[self.train_image_ids_sorted]
        ).float()

        self.train_img_id_to_caption_indices = defaultdict(list)
        train_caption_labels = label_ids_all[is_train_caption]
        for caption_idx, original_img_id in enumerate(train_caption_labels):
            self.train_img_id_to_caption_indices[original_img_id].append(caption_idx)

        print(
            f"Built caption lookup for {len(self.train_image_ids_sorted)} training images."
        )

    def _build_faiss_index(self):
        """
        Build a FAISS index used for hard image sampling.

        :raises RuntimeError: If image structures have not been built yet.
        :raises ImportError: If ``faiss`` is not installed.
        """
        if self.train_images_unique is None:
            raise RuntimeError(
                "Cannot build FAISS index before building image structures."
            )

        print("Building FAISS index for hard image sampling...")

        try:
            import faiss
        except ImportError:
            raise ImportError(
                "Please install faiss-cpu or faiss-gpu to use 'sample_hard_images=True'."
            )

        # Normalize image vectors to unit norm
        embeddings = self.train_images_unique.clone()
        embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
        embeddings = embeddings.cpu().numpy()

        # Build FAISS index using L2 distance
        d = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(d)
        self.faiss_index.add(embeddings)

        print(
            f"FAISS index built (d={d}, N={self.faiss_index.ntotal}) using normalized embeddings."
        )

    def update_batch_mode(self, group_captions: bool, sample_hard_images: bool):
        """
        Switch between random, grouped, and grouped+hard training modes.

        :param group_captions: If ``True``, batch 5 captions per image together.
        :param sample_hard_images: If ``True``, enable FAISS-backed hard sampling.
        :raises ValueError: If invalid combinations or batch sizes are provided.
        """
        # Flip flags
        self.group_captions = group_captions
        self.sample_hard_images = sample_hard_images

        # Ensure valid configuration
        if self.group_captions:
            if self.batch_size % 5 != 0:
                raise ValueError(
                    "When enabling grouped captions, batch_size must be multiple of 5."
                )
            # Always define images_per_batch
            self.images_per_batch = getattr(
                self, "images_per_batch", self.batch_size // 5
            )
        if self.sample_hard_images and not self.group_captions:
            raise ValueError(
                "'sample_hard_images=True' requires 'group_captions=True'."
            )

        if self.group_captions and self.train_image_ids_sorted is None:
            self._build_image_structures(
                self.Y_unique_cached,  # cached in setup()
                self.train_image_ids_cached,
                self.label_ids_all_cached,
                self.is_train_caption_cached,
            )
        if self.sample_hard_images and self.faiss_index is None:
            self._build_faiss_index()

        print(
            f"Switched to mode: group_captions={self.group_captions}, "
            f"sample_hard_images={self.sample_hard_images}"
        )

    @staticmethod
    def _one_hot_to_index(one_hot: np.ndarray) -> np.ndarray:
        """Convert one-hot encoded labels to integer indices (argmax)."""
        return np.argmax(one_hot, axis=1)

    @staticmethod
    def _build_round_robin_view_indices(
        caption_label_ids: np.ndarray,
        gallery_ids_sorted: np.ndarray,
        view_offset: int,
    ) -> np.ndarray:
        """
        Pick one caption index per image using a deterministic round-robin rule.

        :param caption_label_ids: Image id for each validation caption.
        :param gallery_ids_sorted: Sorted validation image ids.
        :param view_offset: Offset controlling which caption is chosen per image.
        :returns: Array of caption indices (one per image) ordered like the gallery.
        :raises ValueError: If an image id has no available captions.
        """

        idxs_per_img: Dict[int, List[int]] = defaultdict(list)
        for idx, img_id in enumerate(caption_label_ids):
            idxs_per_img[img_id].append(idx)

        chosen: List[int] = []
        for img_id in gallery_ids_sorted:
            lst = idxs_per_img.get(img_id, [])
            if not lst:
                raise ValueError(f"No captions found for validation image id {img_id}.")
            chosen.append(lst[view_offset % len(lst)])

        return np.asarray(chosen, dtype=np.int64)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load data, split into train/val/test sets, and build MRR views.

        Optionally concatenates a COCO npz before splitting. Images are
        shuffled once, the first ``val_size`` become the validation gallery,
        and caption views are constructed to align with leaderboard semantics.
        """

        if self._setup_complete:
            print("Data module already set up. Skipping redundant setup.")
            return

        print("Setting up data module...")
        print(f"Loading BASE training data...")
        training_base = np.load(
            self.data_path + "/train/train/train.npz", allow_pickle=True
        )
        X_base = training_base["captions/embeddings"]
        Y_unique_base = training_base["images/embeddings"]
        label_one_hot_base = training_base["captions/label"]
        labels_base = self._one_hot_to_index(label_one_hot_base)
        N_base_images = len(Y_unique_base)

        if self.coco_npz_path:
            print(f"Loading COCO training data...")
            training_coco = np.load(self.coco_npz_path, allow_pickle=True)

            X_coco = training_coco["captions/embeddings"]
            Y_unique_coco = training_coco["images/embeddings"]
            label_one_hot_coco = training_coco["captions/label"]
            labels_coco = self._one_hot_to_index(label_one_hot_coco)
            N_coco_images = len(Y_unique_coco)
            print(
                f"Loaded {len(X_coco)} COCO captions and {N_coco_images} unique images."
            )

            # Combine captions
            X_all = np.concatenate((X_base, X_coco))
            # Combine galleries
            Y_unique = np.concatenate((Y_unique_base, Y_unique_coco))
            # Offset COCO labels (so they point to the 2nd half of the gallery)
            labels_coco_offset = labels_coco + N_base_images
            # Combine labels
            label_ids_all = np.concatenate((labels_base, labels_coco_offset))

            print(
                f"Combined datasets. Total images: {len(Y_unique)}, Total captions: {len(X_all)}"
            )
        else:
            # Fallback if no COCO data is provided
            print("No COCO data provided. Using only base training data.")
            X_all = X_base
            Y_unique = Y_unique_base
            label_ids_all = labels_base

        num_images = Y_unique.shape[0]
        rng = np.random.default_rng(self.seed)

        # Split images into train/val by image id
        image_ids = np.arange(num_images)
        rng.shuffle(image_ids)
        val_image_ids = np.sort(image_ids[: self.val_size])
        train_image_ids = image_ids[self.val_size :]

        # Validation gallery
        self.val_image_gallery = torch.from_numpy(Y_unique[val_image_ids]).float()

        # Caption masks
        is_val_caption = np.isin(label_ids_all, val_image_ids)
        is_train_caption = np.isin(label_ids_all, train_image_ids)

        # Validation
        self.X_val = torch.from_numpy(X_all[is_val_caption]).float()
        self.y_val = torch.from_numpy(Y_unique[label_ids_all[is_val_caption]]).float()
        self.labels_val = torch.from_numpy(label_ids_all[is_val_caption]).long()

        # Training
        self.X_train = torch.from_numpy(X_all[is_train_caption]).float()
        self.y_train = torch.from_numpy(
            Y_unique[label_ids_all[is_train_caption]]
        ).float()
        self.labels_train = torch.from_numpy(label_ids_all[is_train_caption]).long()
        print(
            f"Data split: {self.X_train.shape[0]} train captions, {self.X_val.shape[0]} val captions."
        )

        # Caching for dynamic mode switching
        self.Y_unique_cached = Y_unique
        self.train_image_ids_cached = train_image_ids
        self.label_ids_all_cached = label_ids_all
        self.is_train_caption_cached = is_train_caption

        # Optional: Build image-centric structures for grouped/hard sampling
        if self.group_captions:
            self._build_image_structures(
                Y_unique, train_image_ids, label_ids_all, is_train_caption
            )

        # Test
        print(f"Loading test data...")
        test = np.load(self.data_path + "/test/test/test.clean.npz", allow_pickle=True)
        self.X_test = torch.from_numpy(test["captions/embeddings"]).float()
        self.test_ids = test["captions/ids"]
        print(f"Test data: {self.X_test.shape[0]} captions.")

        # Build MRR views (1 caption per image per view, aligned to gallery)
        # Captions for validation images (keep original order for round-robin indexing)
        val_caption_indices_all = np.where(is_val_caption)[0]
        val_caption_labels_all = label_ids_all[
            val_caption_indices_all
        ]  # original image ids per caption

        # Prepare truncation to (folds * batch_size) images
        B = self.val_mrr_batch_size
        N_full = len(val_image_ids)  # total gallery images available for val
        N_used = min(N_full, self.val_mrr_folds * B)
        if N_used < B:
            raise ValueError(
                f"Not enough validation images to build at least one MRR batch "
                f"(need >= {B}, have {N_full})."
            )

        # Truncated gallery used for MRR (same for all views)
        mrr_gallery_ids = val_image_ids[:N_used]
        self.val_image_gallery_mrr = self.val_image_gallery[:N_used].contiguous()

        # Build each view deterministically via round-robin over caption occurrences
        self.X_val_mrr_views = []
        for v in range(self.val_num_caption_views):
            # Select one caption index per image id (ordered by gallery)
            choose_local = self._build_round_robin_view_indices(
                caption_label_ids=val_caption_labels_all,
                gallery_ids_sorted=mrr_gallery_ids,
                view_offset=v,
            )
            # Map back to absolute caption indices in X_all
            chosen_caption_indices = val_caption_indices_all[choose_local]
            X_view = torch.from_numpy(X_all[chosen_caption_indices]).float()
            self.X_val_mrr_views.append(X_view.contiguous())

        # Optional: build relative embeddings (if num_anchors > 0)
        if self.num_anchors > 0:
            print(f"Computing relative embeddings with {self.num_anchors} anchors...")

            # Pick anchors from training captions (before standardization)
            rng_rel = np.random.default_rng(self.seed)
            anchor_idx = rng_rel.choice(
                self.X_train.shape[0], self.num_anchors, replace=False
            )
            anchors = self.X_train[anchor_idx].clone()
            anchors = anchors / anchors.norm(dim=1, keepdim=True)

            # Helper to compute relative embeddings for any tensor
            def _compute_rel(x: torch.Tensor) -> torch.Tensor:
                x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
                return x_norm @ anchors.T

            # Compute for each split
            self.X_train_rel = _compute_rel(self.X_train)
            self.X_val_rel = _compute_rel(self.X_val)
            self.X_test_rel = _compute_rel(self.X_test)
            self.X_val_mrr_views_rel = [_compute_rel(x) for x in self.X_val_mrr_views]

            # Concatenate absolute + relative
            self.X_train = torch.cat([self.X_train, self.X_train_rel], dim=1)
            self.X_val = torch.cat([self.X_val, self.X_val_rel], dim=1)
            self.X_test = torch.cat([self.X_test, self.X_test_rel], dim=1)
            self.X_val_mrr_views = [
                torch.cat([x_abs, x_rel], dim=1)
                for x_abs, x_rel in zip(self.X_val_mrr_views, self.X_val_mrr_views_rel)
            ]

            print(
                f"Relative embeddings added. New input dim = "
                f"{self.X_train.shape[1]} (abs + {self.num_anchors})"
            )

        # Standardization (optional)
        if self.standardize:
            print("Applying standardization (fit on training captions/images)...")
            self.mu_x = self.X_train.mean(dim=0, keepdim=True)
            self.std_x = self.X_train.std(dim=0, keepdim=True) + 1e-8
            self.mu_y = self.y_train.mean(dim=0, keepdim=True)
            self.std_y = self.y_train.std(dim=0, keepdim=True) + 1e-8

            # Training
            self.X_train = (self.X_train - self.mu_x) / self.std_x
            self.y_train = (self.y_train - self.mu_y) / self.std_y

            # Validation
            self.X_val = (self.X_val - self.mu_x) / self.std_x
            self.y_val = (self.y_val - self.mu_y) / self.std_y

            # MRR views + gallery used for MRR
            self.X_val_mrr_views = [
                (x - self.mu_x) / self.std_x for x in self.X_val_mrr_views
            ]
            self.val_image_gallery_mrr = (
                self.val_image_gallery_mrr - self.mu_y
            ) / self.std_y

            # Test
            self.X_test = (self.X_test - self.mu_x) / self.std_x

            # Standardize unique train images (for hard sampling)
            if self.train_images_unique is not None:
                self.train_images_unique = (
                    self.train_images_unique - self.mu_y
                ) / self.std_y

        # Optional: Build FAISS index for hard negative sampling
        if self.sample_hard_images:
            self._build_faiss_index()

        print(
            f"Data setup complete. "
            f"MRR config -> batch_size={self.val_mrr_batch_size}, folds={self.val_mrr_folds}, "
            f"views={self.val_num_caption_views}, N_used={N_used}"
        )
        self._setup_complete = True

    def _build_val_mrr_mixed_equal(self) -> DataLoader:
        """
        Build a synthetic validation loader for mixed-equal MRR evaluation.

        :returns: DataLoader whose items are pre-batched caption tensors for MRR.
        :raises ValueError: If fewer than 5 caption views are available.
        """
        if self.val_num_caption_views < 5:
            raise ValueError(
                f"Mixed-equal MRR requires 5 views (have {self.val_num_caption_views})."
            )

        B = self.val_mrr_batch_size
        V = self.val_num_caption_views  # must be 5 here
        total_chunks = self.val_mrr_folds

        mixed_batches = []
        for fold in range(total_chunks):
            start = fold * B
            rows = []
            # For each of the B images in this fold, pick ONE caption:
            # choose the view by round-robin so each view contributes B/V items.
            for i in range(B):
                v = i % V  # 0..4 cycling, 20 per view when B=100
                # Take the caption for the same image position from that view
                rows.append(self.X_val_mrr_views[v][start + i].unsqueeze(0))
            x_batch = torch.cat(rows, dim=0)
            mixed_batches.append(x_batch)

        # Pre-batched list: each item is already a full batch (B, 1024)
        ds_mixed = [(x,) for x in mixed_batches]
        return DataLoader(ds_mixed, batch_size=None, shuffle=False, num_workers=0)

    # Dataloaders
    def train_dataloader(self) -> DataLoader:
        """
        Build the training dataloader aligned with the selected batching mode.

        :returns: DataLoader yielding ``(X_train, y_train, labels_train)`` tuples.
        """
        ds = TensorDataset(self.X_train, self.y_train, self.labels_train)

        if not self.group_captions:
            # Default (no grouping)
            print("Using default shuffled DataLoader for training.")
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )

        if self.sample_hard_images:
            # Grouping + Images Hard Mining
            print("Using HardImageBatchSampler for training.")
            batch_sampler = HardImageBatchSampler(
                faiss_index=self.faiss_index,
                train_images_unique=self.train_images_unique,
                train_image_ids_sorted=self.train_image_ids_sorted,
                img_id_to_caption_indices=self.train_img_id_to_caption_indices,
                images_per_batch=self.images_per_batch,
            )
        else:
            # Grouping Only
            print("Using GroupedImageBatchSampler for training.")
            batch_sampler = GroupedImageBatchSampler(
                train_image_ids_sorted=self.train_image_ids_sorted,
                img_id_to_caption_indices=self.train_img_id_to_caption_indices,
                images_per_batch=self.images_per_batch,
                shuffle=True,
            )

        # When using batch_sampler, batch_size must be None and shuffle must be False.
        # The batch_sampler handles both batching and shuffling.
        return DataLoader(
            ds, batch_sampler=batch_sampler, num_workers=0, pin_memory=True
        )

    def val_dataloader(self) -> List[DataLoader]:
        """
        Build validation dataloaders for loss and MRR computation.

        :returns: List whose first element yields ``(X_val, y_val, labels_val)``,
            followed by view-aligned caption loaders for MRR (and an optional mixed view).
        """
        # Validation loss loader
        ds_val = TensorDataset(self.X_val, self.y_val, self.labels_val)
        val_loss_loader = DataLoader(
            ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # MRR loaders
        mrr_loaders: List[DataLoader] = []
        for v, X_view in enumerate(self.X_val_mrr_views, start=1):
            ds_view = TensorDataset(X_view)
            loader_view = DataLoader(
                ds_view,
                batch_size=self.val_mrr_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=False,
            )
            mrr_loaders.append(loader_view)

        # Mixed-equal MRR loader (if 5 views available)
        try:
            mixed_loader = self._build_val_mrr_mixed_equal()
            mrr_loaders.append(mixed_loader)
        except ValueError:
            print("Skipping mixed-equal MRR loader (need 5 views).")

        return [val_loss_loader] + mrr_loaders

    def predict_dataloader(self) -> DataLoader:
        """
        Prediction (test) loader.

        :returns: DataLoader yielding ``(X_test,)``.
        """
        ds_pred = TensorDataset(self.X_test)
        return DataLoader(
            ds_pred,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Mirror of ``predict_dataloader`` for Lightning test loops.

        :returns: DataLoader yielding ``(X_test,)``.
        """
        ds_test = TensorDataset(self.X_test)
        return DataLoader(
            ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
