from abc import ABC, abstractmethod

class Metric(ABC):
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def update(self, y_true, y_pred):
        raise NotImplementedError("Subclasses must implement update method")
        
    @abstractmethod
    def compute(self):
        raise NotImplementedError("Subclasses must implement compute method")