from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from torch import nn

class EHRDrecModel(ABC):
    
    @classmethod
    @abstractmethod
    def requirements(cls):
        """
        Declare the data and side information required by the model.
        """
        ...
    
    @abstractmethod
    def fit(
        self,
        train_data: Any,
        validation_data: Any,
        resources: dict[str, Any] | None = None,
    ) -> None:
        ...
    
    @abstractmethod
    def predict(
        self,
        data: Any,
        resources: dict[str, Any] | None = None,
    ) -> Any:
        ...
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save the model to the specified path.
        """
        ...
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Load the model from the specified path.
        """
        ...

class TorchEHRDrecModel(EHRDrecModel, nn.Module):
    def __init__(self):
        nn.Module.__init__(self)