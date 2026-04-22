from ehrdrec.loading import MIMIC3Loader

if __name__ == "__main__":
    loader = MIMIC3Loader()
    data = loader.load("/home/cararc/data/mimic-iii-1.4")
    print(data)