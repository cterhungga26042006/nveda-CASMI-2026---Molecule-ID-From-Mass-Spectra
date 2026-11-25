
import sys
from pathlib import Path
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# Make local lupin modules importable.
LUPIN_DIR = str(Path(__file__).resolve().parent)
if LUPIN_DIR not in sys.path:
    sys.path.insert(0, LUPIN_DIR)

from candidate_features import extract_molecular_features


@lru_cache(maxsize=250000)
def extract_molecular_features_v2(candidate_smiles):
    """Baseline descriptors plus 12 additional structural descriptors."""

    features = dict(extract_molecular_features(candidate_smiles))

    mol = Chem.MolFromSmiles(str(candidate_smiles))
    if mol is None:
        raise ValueError(f"Invalid candidate SMILES: {candidate_smiles}")

    aromatic_atoms = sum(atom.GetIsAromatic() for atom in mol.GetAtoms())
    heavy_atoms = max(mol.GetNumHeavyAtoms(), 1)

    features.update({
        "v2_fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "v2_aromatic_atoms": int(aromatic_atoms),
        "v2_hetero_atoms": int(sum(
            atom.GetAtomicNum() not in (1, 6)
            for atom in mol.GetAtoms()
        )),
        "v2_formal_charge": int(Chem.GetFormalCharge(mol)),
        "v2_stereocenters": int(len(
            Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        )),
        "v2_fraction_aromatic_atoms": float(aromatic_atoms / heavy_atoms),
        "v2_molecular_complexity": float(Descriptors.BertzCT(mol)),
        "v2_num_aliphatic_rings": int(
            rdMolDescriptors.CalcNumAliphaticRings(mol)
        ),
        "v2_num_saturated_rings": int(
            rdMolDescriptors.CalcNumSaturatedRings(mol)
        ),
        "v2_num_heterocycles": int(
            rdMolDescriptors.CalcNumHeterocycles(mol)
        ),
        "v2_num_amide_bonds": int(
            rdMolDescriptors.CalcNumAmideBonds(mol)
        ),
        "v2_num_bridgehead_atoms": int(
            rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        ),
    })

    return features
