from pathlib import Path
import pandas as pd

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.inchi import MolToInchiKey

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

OUT.mkdir(parents=True, exist_ok=True)

# Change these names to match the downloaded CSV files.
# The smiles column is the only required column.
SOURCES = [
    {
        "name": "COCONUT",
        "path": RAW / "coconut" / "coconut.csv",
        "smiles": "smiles",
        "id": "id",
    },
    {
        "name": "LOTUS",
        "path": RAW / "lotus" / "lotus.csv",
        "smiles": "smiles",
        "id": "id",
    },
    {
        "name": "NPAtlas",
        "path": RAW / "npatlas" / "npatlas.csv",
        "smiles": "smiles",
        "id": "id",
    },
]

enumerator = rdMolStandardize.TautomerEnumerator()


def process_smiles(smiles):
    if pd.isna(smiles) or not isinstance(smiles, str):
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    # Example standardization policy.
    # Must be checked against the actual competition metric.py.
    mol = enumerator.Canonicalize(mol)
    Chem.RemoveStereochemistry(mol)

    canonical = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=False,
    )

    key = MolToInchiKey(mol)

    if not key:
        return None

    return {
        "canonical_smiles": canonical,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "exact_mass": Descriptors.ExactMolWt(mol),
        "inchikey14": key[:14],
    }


def main():
    records = []

    for source in SOURCES:
        path = source["path"]

        if not path.exists():
            raise FileNotFoundError(
                f"Missing source file: {path}\n"
                "Check the filename and folder."
            )

        df = pd.read_csv(path, low_memory=False)

        smiles_col = source["smiles"]
        id_col = source["id"]

        if smiles_col not in df.columns:
            raise ValueError(
                f"{path.name}: column '{smiles_col}' not found.\n"
                f"Available columns: {df.columns.tolist()}"
            )

        print(f"Processing {source['name']}: {len(df):,} rows")

        for _, row in df.iterrows():
            result = process_smiles(row[smiles_col])

            if result is None:
                continue

            original_id = (
                str(row[id_col])
                if id_col in df.columns and pd.notna(row[id_col])
                else ""
            )

            records.append({
                **result,
                "source_database": source["name"],
                "source_id": original_id,
            })

    if not records:
        raise RuntimeError("No valid molecules were processed.")

    molecules = pd.DataFrame(records)

    before = len(molecules)

    # Collapse structures that share the canonical InChIKey14.
    molecules = molecules.drop_duplicates(
        subset=["inchikey14"],
        keep="first",
    ).reset_index(drop=True)

    molecules.insert(
        0,
        "molecule_id",
        [f"M{i:08d}" for i in range(1, len(molecules) + 1)],
    )

    molecules.to_parquet(
        OUT / "molecule_database.parquet",
        index=False,
    )

    molecules.to_csv(
        OUT / "molecule_database.csv",
        index=False,
    )

    print("\nDatabase preparation complete.")
    print("RDKit version:", rdBase.rdkitVersion)
    print(f"Valid structures: {before:,}")
    print(f"Unique structures: {len(molecules):,}")
    print(f"Removed duplicates: {before - len(molecules):,}")
    print("Saved:", OUT / "molecule_database.parquet")


if __name__ == "__main__":
    main()