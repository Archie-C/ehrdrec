# ehrdrec

## Data

### MIMIC-III and MIMIC-IV

### Creating DDI Mappings

We use the datasets from [here](https://ddinter2.scbdd.com/download/) to get drug-drug-interactions with severity. We then used `ddi_inter_scraper.py` to fill create a DDI inter code to ATC level 5 mapping. This isn't exhaustive so the rest were filled in by hand where possible using https://go.drugbank.com/. The resulting file is `data/ddinter2/mapping/ddinter_atc_codes.csv`. We then map ATC codes to interactions and severity using `create_ddinter_mapping.py` which creates `data/ddinter2/mapping/ddinter_mapped_atc_codes.csv`

### Creating SMILES Mapping
Due to the inavailability of the research downloads from DrugBank [here](https://go.drugbank.com/releases/latest?_gl=1*1tk8okz*_up*MQ..*_ga*ODg3MjQ1NDQuMTc4NzAzNjU3OA..*_ga_DDLJ7EEV9M*czE3ODcwMzY1NzckbzEkZzEkdDE3ODcwMzc4NjgkajQ4JGwwJGgw) at the time of writing this, we created the mapping `drugbank_atc_smiles.json` using first `drugbank_atc_scraper.py` and then `drugbank_atc_smiles_scraper.py`. We provide these in the `utils` folder at the root. And the data files can be found under `data/drugbank`.

> Note that some ATC Level 5 codes are missing DrugBank drugs, and some of these are missing SMILES, these are documented in `drugbank_atc_drugs.errors.json` and `drugbank_atc_smiles.errors.json`

