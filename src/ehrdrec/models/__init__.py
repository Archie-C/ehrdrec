from contextlib import suppress

with suppress(ModuleNotFoundError):
    from .torch.mlp import MLP
with suppress(ModuleNotFoundError):
    from .torch.GAMENet import GameNetFast
with suppress(ModuleNotFoundError):
    from .torch.foursdrug import FourSDrug
with suppress(ModuleNotFoundError):
    from .torch.FastRx import FastRx
with suppress(ModuleNotFoundError):
    from .torch.micron import Micron
with suppress(ModuleNotFoundError):
    from .torch.original.mr_dtr import MRDTR
with suppress(ModuleNotFoundError):
    from .torch.original.rpnet import RPNet

from .dataclasses import (
    Medication,
    ExtendedMedication,
    LoadedData,
    ProcessedData,
    ProcessedDataMultiHot,
    ProcessedEHRSequence,
    ExperimentConfig,
    TrainingResults,
    EvaluationResults,
)

__all__ = [
    name
    for name in [
        "MLP",
        "GameNetFast",
        "FourSDrug",
        "FastRx",
        "Micron",
        "Medication",
        "ExtendedMedication",
        "LoadedData",
        "ProcessedData",
        "ProcessedDataMultiHot",
        "ProcessedEHRSequence",
        "ExperimentConfig",
        "TrainingResults",
        "EvaluationResults",
        "MRDTR",
        "RPNet",
    ]
    if name in globals()
]
