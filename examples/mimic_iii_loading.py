import logging

from ehrdrec.loading import MIMIC3Loader

logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()

if __name__ == "__main__":
    loader = MIMIC3Loader()
    data = loader.load("/home/cararc/data/mimic-iii-1.4")