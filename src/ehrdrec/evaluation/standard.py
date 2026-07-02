import torch

from ehrdrec.evaluation import BaseEvaluator
from ehrdrec.metrics import Metric
from ehrdrec.models.dataclasses import EvaluationResults


class Evaluator(BaseEvaluator):
    def __init__(
        self,
        model,
        test_loader,
        metrics: list[Metric] = None,
        device: str = "cuda",
        save_predictions: bool = False,
    ):
        super().__init__(model, test_loader, metrics, device)
        self.save_predictions = save_predictions

    def run(self) -> EvaluationResults:
        if not self.metrics:
            raise ValueError("No metrics specified for evaluation.")
        for metric in self.metrics:
            metric.reset()
        self.model.eval()

        all_predictions: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in self.test_loader:
                inputs, targets = batch
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(self.device)
                targets = targets["atc5"].to(self.device) if isinstance(targets, dict) else targets.to(self.device)

                output = self.model(inputs)
                logits = output["predictions"] if isinstance(output, dict) else output

                for metric in self.metrics:
                    metric.update(logits, targets)

                if self.save_predictions:
                    all_predictions.append(torch.sigmoid(logits).cpu())
                    all_targets.append(targets.cpu())

        test_metrics = {metric.name: metric.compute() for metric in self.metrics}

        predictions = torch.cat(all_predictions, dim=0) if self.save_predictions else None
        targets_out = torch.cat(all_targets, dim=0) if self.save_predictions else None

        return EvaluationResults(
            test_metrics=test_metrics,
            predictions=predictions,
            targets=targets_out,
        )
