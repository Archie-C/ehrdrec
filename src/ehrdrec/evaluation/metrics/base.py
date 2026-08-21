
from abc import ABC, abstractmethod

from ehrdrec.evaluation.contracts import EvaluationOutput


class Metric(ABC):
    def __init__(self, name: str, threshold: float = 0.5):
        self.name = name
        self.threshold = threshold

    @abstractmethod
    def compute(self, output: EvaluationOutput) -> dict[str, float]:
        raise NotImplementedError("Subclasses must implement the compute method.")