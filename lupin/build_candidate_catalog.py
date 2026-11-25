
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

INPUT = "outputs/ranking_train.parquet"
OUTPUT = "outputs/train_candidate_catalog.parquet"

catalog = {}
invalid = 0
formula_mismatches = 0

print("Reading training structures...")

parquet = pq.ParquetFile(INPUT)

for batch in parquet.iter_batches(
    batch_size=50000,
    columns=[
        "normalized_smiles",
        "inchikey14",
        "molecular_formula",
    ],
):
    smiles_list = batch.column("normalized_smiles").to_pylist()
    keys = batch.column("inchikey14").to_pylist()
    formulas = batch.column("molecular_formula").to_pylist()

    for smiles, key, formula in zip(smiles_list, keys, formulas):
        if not smiles or not key or not formula:
            continue

        if key in catalog:
            continue

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            invalid += 1
            continue

        calculated_formula = rdMolDescriptors.CalcMolFormula(mol)

        if calculated_formula != formula:
            formula_mismatches += 1
            continue

        catalog[key] = {
            "inchikey14": key,
            "smiles": smiles,
            "molecular_formula": formula,
        }

print("\nCATALOG RESULTS")
print("=" * 45)
print(f"Unique valid structures: {len(catalog):,}")
print(f"Invalid SMILES:          {invalid:,}")
print(f"Formula mismatches:      {formula_mismatches:,}")

table = pa.Table.from_pylist(list(catalog.values()))
pq.write_table(table, OUTPUT, compression="zstd")

print(f"\nSaved: {OUTPUT}")
print(f"Catalog records: {table.num_rows:,}")
