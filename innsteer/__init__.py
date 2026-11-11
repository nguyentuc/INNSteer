"""INNSteer: invertible latent transformations for activation steering (NeurIPS 2026)."""
from .inn import INN
from .steering import innsteer_hooks, innsteer_transform, latent_steering_vector
from .train import (
    DEFAULT_INN_CFG,
    DEFAULT_TRAIN_CFG,
    split_activations,
    train_best_inn_for_layer,
    train_inn_for_layer,
    zscore_normalize_eval,
    zscore_normalize_train,
)

__version__ = "1.0.0"
