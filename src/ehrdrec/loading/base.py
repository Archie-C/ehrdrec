
from abc import ABC, abstractmethod

class BaseLoader(ABC):
    
    @abstractmethod
    def load(self, source: str):
        """Load data from the specified source and then validate it."""
        raise NotImplementedError("Subclasses must implement the load method.")
