import logging

from ehrdrec.loading import MIMIC4Loader

logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()

if __name__ == "__main__":
    loader = MIMIC4Loader()
    data = loader.load("/home/cararc/data/mimic-iv-3.1/hosp")
    print(data.frame.collect().head())
