import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem

print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("RDKit:", rdkit.__version__)

mol = Chem.MolFromSmiles("CCO")

if mol is None:
    raise RuntimeError("RDKit cannot parse the test molecule")

print("SMILES test:", Chem.MolToSmiles(mol))
print("Environment is working!")