# src/models/load_model.py

from torch import nn
import torch.nn.functional as F
from src.models.mlp_connector import MlpConnector

from src.models.models_others import (
    MLPTranslator,
    DeepTranslator,
    TranslatorModelV2,
    AdaLNTranslator,
)


def build_model_from_config(cfg: dict) -> nn.Module:
    """
    Instantiate a translator model based on a configuration mapping.

    :param cfg: Configuration dictionary containing a ``type`` key plus any
        keyword arguments accepted by the target model constructor.
    :returns: Concrete ``nn.Module`` matching the requested ``type``.
    :raises ValueError: If ``cfg['type']`` is not a supported model.

    Example
    -------
    ``model = build_model_from_config(config[\"model\"])``
    """
    model_type = cfg.get("type", "MLPTranslator").lower()
    kwargs = {k: v for k, v in cfg.items() if k != "type"}

    if model_type == "mlptranslator":
        return MLPTranslator(**kwargs)
    elif model_type == "deeptranslator":
        return DeepTranslator(**kwargs)
    elif model_type in {"translatorv2", "translatormodelv2"}:
        return TranslatorModelV2(**kwargs)
    elif model_type.lower() == "adalntranslator":
        return AdaLNTranslator(**{k: v for k, v in cfg.items() if k != "type"})
    elif model_type.lower() == "mlpconnector":
        return MlpConnector(**{k: v for k, v in cfg.items() if k != "type"})

    else:
        raise ValueError(f"Unknown model type: {cfg.get('type')}")
