import torch

from ehrdrec.evaluation import BaseEvaluator
from ehrdrec.metrics import Metric
from ehrdrec.models.dataclasses import EvaluationResults

class Evaluator(BaseEvaluator):
    def __init__(
        self,
        model,
        test_loader,
        metrics:list[Metric]=None,
        device="cuda",
    ):
        super().__init__(model, test_loader, metrics, device)
    
    def run(self) -> EvaluationResults:
        if not self.metrics:
            raise ValueError("No metrics specified for evaluation.")
        for metric in self.metrics:
            metric.reset()
        self.model.eval()

        with torch.no_grad():
            for batch in self.test_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                for metric in self.metrics:
                    metric.update(outputs, targets)

        # Average metrics over all batches
        test_metrics = {metric.name: metric.compute() for metric in self.metrics}

        return EvaluationResults(test_metrics=test_metrics)