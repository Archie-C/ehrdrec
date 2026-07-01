import torch

from ehrdrec.evaluation import BaseEvaluator
from ehrdrec.metrics import Metric
from ehrdrec.models.dataclasses import EvaluationResults


class LLMEvaluator(BaseEvaluator):
    """
    Evaluator for LLM recommendation adapters.

    Expects batches shaped like ``(list[dict], Tensor[batch, n_medications])``.
    Model outputs should contain binary/probability ``predictions`` aligned with
    the medication vocabulary, plus optional generated ``texts`` and ``atc_codes``.
    """

    def __init__(
        self,
        model,
        test_loader,
        metrics: list[Metric] = None,
        device: str = "cpu",
        save_predictions: bool = False,
        save_generations: bool = False,
    ):
        self.device = torch.device(device)
        self.model = model
        self.test_loader = test_loader
        self.metrics = metrics or []
        self.save_predictions = save_predictions
        self.save_generations = save_generations
        self.generations: list[dict] = []

    def run(self) -> EvaluationResults:
        if not self.metrics:
            raise ValueError("No metrics specified for evaluation.")

        for metric in self.metrics:
            metric.reset()

        if hasattr(self.model, "eval"):
            self.model.eval()

        all_predictions: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        self.generations = []

        for features, targets in self.test_loader:
            targets = targets.to(self.device)
            output = self.model(features)
            predictions = output["predictions"].to(self.device)

            for metric in self.metrics:
                metric.update(predictions, targets)

            if self.save_predictions:
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

            if self.save_generations:
                texts = output.get("texts", [])
                responses = output.get("responses", texts)
                raw_responses = output.get("raw_responses", responses)
                prompts = output.get("prompts", [None] * len(texts))
                atc_codes = output.get("atc_codes", [])
                for feature, prompt, text, response, raw_response, codes in zip(
                    features,
                    prompts,
                    texts,
                    responses,
                    raw_responses,
                    atc_codes,
                ):
                    self.generations.append(
                        {
                            "patient_id": feature.get("patient_id"),
                            "admission_id": feature.get("admission_id"),
                            "prompt": prompt,
                            "text": text,
                            "response": response,
                            "raw_response": raw_response,
                            "atc_codes": codes,
                            "n_parsed_atc_codes": len(codes),
                            "target_atc_codes": feature.get("target_atc_codes"),
                        }
                    )

        test_metrics = {metric.name: metric.compute() for metric in self.metrics}

        predictions_out = (
            torch.cat(all_predictions, dim=0) if self.save_predictions else None
        )
        targets_out = torch.cat(all_targets, dim=0) if self.save_predictions else None

        return EvaluationResults(
            test_metrics=test_metrics,
            predictions=predictions_out,
            targets=targets_out,
        )
