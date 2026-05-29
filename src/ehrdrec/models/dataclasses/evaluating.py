
from dataclasses import dataclass

@dataclass(slots=True)
class EvaluationResults:
    test_metrics: dict[str, float]