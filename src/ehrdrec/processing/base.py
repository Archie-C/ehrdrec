
from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    
    @abstractmethod
    def process(self, data):
        """Process the loaded data."""
        raise NotImplementedError("Subclasses must implement the process method.")
