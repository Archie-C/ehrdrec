import torch

from ehrdrec.evaluation.metrics import Jaccard, F1, PRAUC, LRAP
from ehrdrec.experiments import ExperimentRunner
from ehrdrec.models import RETAIN
from ehrdrec.tasks import MedicationSetRecommendationTask
from ehrdrec.tasks.medication_set_recommendation.task import MedicationSplitType
from ehrdrec.training import TrainerConfig

config = {
    "mimic3_path": "/home/cararc/data/mimic-iii-1.4",
    "ndc_atc_mapping_file": "data/mappings/ndc_atc_mapping.sqlite",
    "atc_level": 3,
    "split_type": MedicationSplitType.LAST_VISIT,
}


def main():
    
    task = MedicationSetRecommendationTask(config=config)

    runner = ExperimentRunner(
        output_root="artifacts",
        trainer_config=TrainerConfig(
            epochs=10,
            selection_metric="jaccard",
            selection_mode="max",
        ),
        metrics=[Jaccard(), F1(), PRAUC(), LRAP()],
        batch_size=32,
    )

    experiment = runner.run(
        model=RETAIN,
        task=task,
        seeds=[42, 123, 456],
        model_config={"embedding_dim": 128},
        optimizer_factory=lambda parameters: torch.optim.Adadelta(parameters),
    )


if __name__ == "__main__":
    main()