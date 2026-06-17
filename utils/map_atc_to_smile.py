
from chembl_webresource_client.new_client import new_client

molecule = new_client.molecule

def atc_to_smile(atc_code):
    results = molecule.filter(atc_classifications__level5=atc_code)
    if results:
        return results[0]['molecule_structures']['canonical_smiles']
    return None

smiles_dict = {}
for atc in your_atc_codes:
    smiles_dict[atc] = atc_to_smiles(atc)