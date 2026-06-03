import time
import logging
import pickle
import os

from ehrdrec.datasets.multi_hot import MultiHotDataset
from ehrdrec.evaluation import Evaluator
from ehrdrec.loading import MIMIC3Loader
from ehrdrec.metrics import F1, Jaccard, PRAUC, BinaryDDI
from ehrdrec.metrics.ddi import HighSeverityBinaryDDI
from ehrdrec.processing import MultiHotProcessor
from ehrdrec.training import Trainer, ConsoleLogger, CheckpointLogger, CompositeLogger
from ehrdrec.training.losses import BCELoss
from ehrdrec.models import MLP
import torch

from torch.utils.data import DataLoader

from chembl_webresource_client.new_client import new_client

molecule = new_client.molecule

def atc_to_smiles(atc_code):
    try:
        results = molecule.filter(atc_classifications__level5=atc_code)
        if results:
            smiles = results[0]['molecule_structures']['canonical_smiles']
            return smiles
    except Exception as e:
        print(f"Failed for {atc_code}: {e}")
    return None

logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()

ATC_LEVEL = 5
SMILES_SAVE_PATH = "atc2smiles.pkl"

if __name__ == "__main__":
    loader = MIMIC3Loader()
    data = loader.load("/home/cararc/data/mimic-iii-1.4")
    processor = MultiHotProcessor()
    processed_data = processor.process(data, minimum_admissions=2, atc_level=ATC_LEVEL, force_reload=True)
    medications_vocab = processor.medications_vocab

    # Load existing progress if available
    if os.path.exists(SMILES_SAVE_PATH):
        with open(SMILES_SAVE_PATH, 'rb') as f:
            smiles_dict = pickle.load(f)
        print(f"Loaded existing SMILES dict with {len(smiles_dict)} entries")
    else:
        smiles_dict = {}

    all_atc_codes = list(medications_vocab.id_to_token.values())
    
    for atc in all_atc_codes:
        # Skip if already fetched
        if atc in smiles_dict:
            print(f"Skipping {atc} (already fetched)")
            continue
        
        smiles = atc_to_smiles(atc)
        smiles_dict[atc] = smiles
        print(f"{atc}: {smiles}")
        time.sleep(0.1)

        # Save after every fetch in case of interruption
        with open(SMILES_SAVE_PATH, 'wb') as f:
            pickle.dump(smiles_dict, f)

    missing = [atc for atc, smi in smiles_dict.items() if smi is None]
    print(f"\nMissing: {len(missing)}/{len(all_atc_codes)}")
    print(missing)