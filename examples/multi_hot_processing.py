import logging

from ehrdrec.loading import MIMIC3Loader
from ehrdrec.processing import MultiHotProcessor

logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()

if __name__ == "__main__":
    loader = MIMIC3Loader()
    data = loader.load("/home/cararc/data/mimic-iii-1.4")
    processor = MultiHotProcessor()
    processed_data = processor.process(data, minimum_admissions=2)
    print("Train data:")
    print(processed_data.train_frame.collect().head())
    print("Validation data:")
    print(processed_data.val_frame.collect().head())
    print("Test data:")
    print(processed_data.test_frame.collect().head())