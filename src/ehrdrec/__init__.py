# Seeding / run persistence
from .utils import seed_everything, save_run

# Loading
from .loading import MIMIC3Loader, MIMIC4Loader, BaseLoader

# Processing
from .processing import MultiHotProcessor, MultiHotProcessorAllATCs, SetSequenceProcessor

# Datasets
from .datasets import (
    MultiHotDataset,
    MultiHotDatasetWithPatientLookBack,
    MultiHotDatasetWithAllATCLevels,
    OriginalGAMENetDataset,
    collate_patient_visit_histories,
    collate_original_gamenet,
    MRDTRBatch,
    MRDTRDataset,
    build_mrdtr_graph,
    collate_mrdtr_examples,
)

# Models
from . import models as _models

for _model_name in ("MLP", "GameNetFast", "FourSDrug", "FastRx", "Micron", "RPNet"):
    if hasattr(_models, _model_name):
        globals()[_model_name] = getattr(_models, _model_name)

# Data contracts
from .models import (
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

# Training
from .training import Trainer, Tuner, StagedJepaTrainer, FreezeConfig, OriginalGAMENetTrainer

# Evaluation
from .evaluation import Evaluator, LLMEvaluator

# Metrics
from .metrics import F1, Jaccard, PRAUC, BinaryDDI, HighSeverityBinaryDDI

# Mappings
from .mappings import (
    NDCATCMapper,
    MappingBuilder,
    SQLiteMappingStore,
    Vocab,
    ATCMapping,
    MappingResult,
    normalise_ndc,
    atc_to_level,
    NDCATCError,
    InvalidNDCError,
    MappingStoreError,
    MappingNotFoundError,
)

__all__ = [
    # Seeding / run persistence
    "seed_everything",
    "save_run",
    # Loading
    "MIMIC3Loader",
    "MIMIC4Loader",
    "BaseLoader",
    # Processing
    "MultiHotProcessor",
    "MultiHotProcessorAllATCs",
    "SetSequenceProcessor",
    # Datasets
    "MultiHotDataset",
    "MultiHotDatasetWithPatientLookBack",
    "MultiHotDatasetWithAllATCLevels",
    "OriginalGAMENetDataset",
    "collate_patient_visit_histories",
    "collate_original_gamenet",
    "MRDTRBatch",
    "MRDTRDataset",
    "build_mrdtr_graph",
    "collate_mrdtr_examples",
    # Models
    # Optional torch models
    # Data contracts
    "Medication",
    "ExtendedMedication",
    "LoadedData",
    "ProcessedData",
    "ProcessedDataMultiHot",
    "ProcessedEHRSequence",
    "ExperimentConfig",
    "TrainingResults",
    "EvaluationResults",
    # Training
    "Trainer",
    "Tuner",
    "StagedJepaTrainer",
    "FreezeConfig",
    "OriginalGAMENetTrainer",
    # Evaluation
    "Evaluator",
    "LLMEvaluator",
    # Metrics
    "F1",
    "Jaccard",
    "PRAUC",
    "BinaryDDI",
    "HighSeverityBinaryDDI",
    # Mappings
    "NDCATCMapper",
    "MappingBuilder",
    "SQLiteMappingStore",
    "Vocab",
    "ATCMapping",
    "MappingResult",
    "normalise_ndc",
    "atc_to_level",
    "NDCATCError",
    "InvalidNDCError",
    "MappingStoreError",
    "MappingNotFoundError",
]

__all__.extend(
    name
    for name in ["MLP", "GameNetFast", "FourSDrug", "FastRx", "Micron", "RPNet"]
    if name in globals()
)
