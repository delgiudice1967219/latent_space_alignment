# AML Kaggle Competition (2025-2026) - Group: Golden retrieval

This repository contains the source code for the "Golden retrieval" group's submission to the **Advanced Machine Learning** course Kaggle competition (2025-2026).

## Group Members

* Valeria Avino (ID: `1905974`)
* Xavier Del Giudice (ID: `1967219`)
* Gabriel Pinos (ID: `1965035`)
* Leonardo Rocci (ID: `1922496`)

---

## Project Overview

This project provides a modular and configurable pipeline for training a model to translate from a caption-embedding space to an image-embedding space. The code is organized into a `src` directory and demonstrated in the `main.ipynb` notebook, which contains our final, best-performing experiment.

## Project Structure

The project is organized within the `src` directory, which contains all the core logic. The final submission notebook (`opt_1.ipynb`) imports from these modules.

```
.
├── src/
│   ├── data.py
│   ├── losses.py
│   ├── trainer.py
│   ├── utils.py
│   ├── models/
│   │   ├── load_model.py
│   │   └── mlp_connector.py
│
├── main.ipynb
└── requirements.txt
```

---

## Module Descriptions

Here is a brief overview of each module's responsibility:

### `src/data.py`

This module contains the **`EmbeddingDataModule`**, our main `pl.LightningDataModule`.

* It handles loading the `train.npz`, `test.clean.npz`, and optional `coco_2.npz` files. If COCO data is provided, it merges and shuffles it with the primary training data.

* It implements two custom batch samplers:
    * **`GroupedImageBatchSampler`**: A sampler that ensures all 5 captions for a given image are grouped into the same training batch.
    * **`HardImageBatchSampler`**: A more advanced sampler that builds a **Faiss index** to find and batch semantically similar images together, creating "hard" negative examples.
* It prepares validation data by creating multiple "views" (`val_num_caption_views`), where each view is a complete, aligned set of captions (one per image) used for robust MRR calculation.
* It includes a helper function, `update_batch_mode`, to allow a `BatchModeScheduler` to dynamically change the sampling strategy (e.g., from `Grouped` to `Hard`) mid-training.
* Contains the `save_submission` utility function to format and save the final `Golden_Submission.csv`.

### `src/losses.py`

This module defines the loss functions used in the `TranslationTrainer`.
* **`compute_similarity`**: A flexible function to compute the similarity matrix. It supports standard `dot` product as well as `cosine`, `poly_kernel`, and `rbf_kernel` (with configurable `alpha` and `degree` parameters).

* **Loss Implementations**: Provides implementations for `InfoNCELoss`, `TripletLoss`, and `CircleLoss` (we also tested `FastAP` and `RLL`, altough they are not implemented here).
* **Multi-Positive Support**: The `InfoNCELoss` and `TripletLoss` can accept a `labels` tensor to correctly handle multi-positive examples (i.e., not treating correct captions for the same image as negatives).
* **`LossFactory`**: A simple factory class (`LossFactory.build(name, **kwargs)`) to instantiate the desired loss function by name (e.g., "triplet") from the main config.

### `src/trainer.py`

This module defines the **`TranslationTrainer`**, our main `pl.LightningModule`.
* It encapsulates the complete training, validation, and prediction logic.

* **`compute_loss`**: This is the core logic. It computes the similarity matrix, then passes it to the chosen loss function (`self.loss_fn`). It also handles:
    * **`symmetric`**: If true, it computes the loss on the transposed similarity matrix as well (for the image-to-caption task) and averages the two.
    * **`use_mse`**: If true, it adds an MSE regularization term (weighted by `_lambda_mse`) between the normalized predicted and true embeddings.
* **Schedulers**: In `on_train_epoch_start`, it steps any provided `AlphaScheduler` (for annealing hyperparameters like `_lambda_mse`) and `BatchModeScheduler`.
* **MRR Validation**: `validation_step` implements the full **MRR (Mean Reciprocal Rank)** calculation. It correctly routes batches (`dataloader_idx`) to either the validation loss calculation or the MRR calculation.
* **`configure_optimizers`**: Sets up the **AdamW** optimizer and an optional **CosineAnnealingLR** or **ReduceLROnPlateau** scheduler based on the config.

### `src/utils.py`

A collection of core helper functions and callbacks.
* **`set_seed`**: A utility to set random seeds for `random`, `numpy`, and `torch` for reproducibility.

* **`StopAtEpochCallback`**: A custom `pl.Callback` that cleanly stops the `pl.Trainer` *after* a specified epoch number has completed. This was used in our final run to train for a fixed number of epochs on the full dataset.

---

### `src/models/mlp_connector.py`

Defines the core model architecture used for our submission.

* **`MlpConnector`**: A simple, fully-configurable Multi-Layer Perceptron (MLP) designed to "connect" the caption embedding space (1024 dims) to the image embedding space (1536 dims).

* The architecture is defined by the config, allowing control over `input_dim`, `output_dim`, `hidden_dim`, `dropout_rate`, `activation_fun` (e.g., `nn.SiLU`), and whether to use `layer_norm` or `normalize_out`.

### `src/models/load_model.py`

Contains a factory function to build models from the configuration dictionary.

* **`build_model_from_config`**: This helper function reads the `model` dictionary from the main `CONFIG`, pops the `type` key (e.g., "MlpConnector", "DeepTranslator"), and uses the remaining items as `kwargs` to instantiate the correct model class. This makes our pipeline flexible and easy to experiment with different architectures.

---

## How to Run

1.  **Install Dependencies:**

    All required packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Data:**
    *(The notebook expects data to be present in a local folder path. This section can be updated with data download/setup instructions.)*

3.  **Run the Notebook:**

    Open and run the `main.ipynb` notebook.
    * **Section 3** defines the final, optimized `CONFIG`.
    * **Section 4** executes the `run_experiment(CONFIG)` function to train the model and generate `Golden_Submission.csv`.
    * **Section 5** contains code to load and inspect the `tuning.db` Optuna database to show evidence of our hyperparameter tuning.