from torch.nn.utils.rnn import pad_sequence
import torch

def collate_patient_visit_histories(batch):
    xs, ys = zip(*batch)

    lengths = torch.tensor(
        [x["diagnoses"].shape[0] for x in xs],
        dtype=torch.long,
    )

    diagnoses = pad_sequence(
        [x["diagnoses"] for x in xs],
        batch_first=True,
    )

    procedures = pad_sequence(
        [x["procedures"] for x in xs],
        batch_first=True,
    )

    medication_history = pad_sequence(
        [x["medication_history"] for x in xs],
        batch_first=True,
    )

    y = torch.stack(ys)

    return {
        "diagnoses": diagnoses,
        "procedures": procedures,
        "medication_history": medication_history,
        "lengths": lengths,
    }, y