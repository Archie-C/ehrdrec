import logging

from ehrdrec.datasets.multi_hot import MultiHotDataset
from ehrdrec.loading import MIMIC3Loader
from ehrdrec.metrics import F1, Jaccard, PRAUC
from ehrdrec.processing import MultiHotProcessor
from ehrdrec.training import Trainer
from ehrdrec.models import MLP
import torch

from torch.utils.data import DataLoader

logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()

if __name__ == "__main__":
    loader = MIMIC3Loader()
    data = loader.load("/home/cararc/data/mimic-iii-1.4")
    processor = MultiHotProcessor()
    processed_data = processor.process(data, minimum_admissions=2, atc_level=3)
    train_dataset = MultiHotDataset(processed_data.train_frame.collect(), target_col="medication_multihot", feature_cols=["diagnosis_multihot", "procedure_multihot"])
    val_dataset = MultiHotDataset(processed_data.val_frame.collect(), target_col="medication_multihot", feature_cols=["diagnosis_multihot", "procedure_multihot"])
    test_dataset = MultiHotDataset(processed_data.test_frame.collect(), target_col="medication_multihot", feature_cols=["diagnosis_multihot", "procedure_multihot"])
    x, y = train_dataset[0]
    output_size = y.shape[0]
    input_size = x.shape[0]
    print(f"Input size: {input_size}, Output size: {output_size}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = MLP(input_size=input_size, hidden_sizes=[512, 256], output_size=output_size)
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = [Jaccard(), F1(), PRAUC()]
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=metrics,
        device="cuda" if torch.cuda.is_available() else "cpu",
        epochs=40,
    )
    results = trainer.fit()
    print("Training results:")
    print(results)