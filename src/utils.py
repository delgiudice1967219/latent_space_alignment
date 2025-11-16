# src/utils.py
import random
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


def set_seed(seed: int = 42, show=False) -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs for reproducible runs.

    :param seed: Seed value applied to Python, NumPy, CUDA, and cuDNN.
    :param show: If ``True``, print a confirmation message after seeding.
    :returns: ``None``. Seeds are set in-place for the relevant libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if show:
        print(f"Seed set to {seed}")


def get_device(show=False) -> torch.device:
    """
    Select the best available torch device, preferring CUDA then MPS.

    :param show: If ``True``, log which device was selected.
    :returns: Torch device pointing to ``cuda``, ``mps``, or ``cpu``.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if show:
            print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        if show:
            print("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        if show:
            print("Using CPU")
    return device


def save_submission(ids, embeddings, filename: str = "submission.csv") -> None:
    """
    Write model outputs to a CSV.

    :param ids: Array of test caption identifiers aligned with ``embeddings``.
    :param embeddings: Predicted image-space embeddings as a tensor or ndarray.
    :param filename: Output CSV path for the submission file.
    :returns: ``None``. The submission file is written to ``filename``.

    .. note::
       The competition expects each row to contain a JSON-like string
       representation of the embedding: ``[e1, e2, ..., eD]``.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    records = []
    for i, emb in zip(ids, embeddings):
        emb_json = "[" + ", ".join(f"{x:.17g}" for x in emb.tolist()) + "]"
        records.append({"id": int(i), "embedding": emb_json})

    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    print(f"Submission saved to {filename} ({len(df)} rows)")


class StopAtEpochCallback(Callback):
    """
    Lightning callback that stops training after a specific epoch.

    :param stop_epoch: Epoch number after which training should halt.
    """

    def __init__(self, stop_epoch: int):
        super().__init__()
        self.stop_epoch = stop_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Request trainer shutdown once the target epoch has finished.

        :param trainer: Lightning trainer.
        :param pl_module: Current Lightning module.
        """
        current_epoch_idx = trainer.current_epoch
        if current_epoch_idx == (self.stop_epoch - 1):
            print(
                f"\n StopAtEpochCallback: Reached epoch {self.stop_epoch}. Stop training."
            )
            trainer.should_stop = True
