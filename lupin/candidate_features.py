
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from adducts import neutral_mass_from_mz


@lru_cache(maxsize=250000)
def extract_molecular_features(candidate_smiles):
    """
    Calculate structure-only features once per unique SMILES.
    Results are cached to avoid repeated RDKit calculations.
    """
    mol = Chem.MolFromSmiles(str(candidate_smiles))

    if mol is None:
        raise ValueError(f"Invalid candidate SMILES: {candidate_smiles}")

    formula = rdMolDescriptors.CalcMolFormula(mol)

    features = {
        "candidate_exact_mass": float(Descriptors.ExactMolWt(mol)),
        "candidate_heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "candidate_total_atoms": int(mol.GetNumAtoms()),
        "candidate_rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "candidate_aromatic_rings": int(
            rdMolDescriptors.CalcNumAromaticRings(mol)
        ),
        "candidate_hbd": int(Lipinski.NumHDonors(mol)),
        "candidate_hba": int(Lipinski.NumHAcceptors(mol)),
        "candidate_rotatable_bonds": int(
            Lipinski.NumRotatableBonds(mol)
        ),
        "candidate_tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "candidate_logp": float(Descriptors.MolLogP(mol)),
        "candidate_molecular_weight": float(Descriptors.MolWt(mol)),
    }



    # Formula counts: always create the same columns.
    import re

    atom_counts = {}

    for element, number in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        atom_counts[element] = int(number) if number else 1

    for element in ("C", "H", "N", "O", "S", "P"):
        features[f"candidate_formula_{element.lower()}"] = (
            atom_counts.get(element, 0)
        )

    features["candidate_formula_halogen"] = sum(
        atom_counts.get(element, 0)
        for element in ("F", "Cl", "Br", "I")
    )

    return features


def extract_candidate_spectrum_features(
    candidate_smiles,
    precursor_mz,
    adduct,
):
    """
    Calculate features that depend on both candidate and spectrum.
    """
    molecular_features = extract_molecular_features(candidate_smiles)

    exact_mass = molecular_features["candidate_exact_mass"]

    observed_neutral_mass = neutral_mass_from_mz(
        float(precursor_mz),
        adduct,
    )

    mass_error = (
        (observed_neutral_mass - exact_mass)
        / exact_mass
        * 1_000_000
        if exact_mass > 0 else 999999.0
    )

    return {
        "candidate_mass_error_ppm": float(mass_error),
        "candidate_abs_mass_error_ppm": abs(float(mass_error)),
    }


def extract_candidate_features(
    candidate_smiles,
    precursor_mz,
    adduct,
    mz_values=None,
    intensities=None,
    ionization_mode=None,
):
    """
    Compatibility wrapper for existing scripts.

    Fragment matching is intentionally excluded from the first
    ranking baseline because same-formula candidates share the
    same possible elemental subformula masses.
    """
    features = extract_molecular_features(candidate_smiles)

    features.update(
        extract_candidate_spectrum_features(
            candidate_smiles,
            precursor_mz,
            adduct,
        )
    )

    return features
