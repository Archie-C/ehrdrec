from abc import ABC, abstractmethod

class Metric(ABC):
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def update(self, outputs, targets):
        raise NotImplementedError("Subclasses must implement update method")
        
    @abstractmethod
    def compute(self):
        raise NotImplementedError("Subclasses must implement compute method")
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError("Subclasses must implement reset method")