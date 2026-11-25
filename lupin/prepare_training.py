
import os
import sys

sys.path.insert(0, "src/lupin")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rdkit import Chem
from rdkit.Chem import Descriptors

from adducts import ADDUCTS, neutral_mass_from_mz, ppm_error


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRAIN_PATH = "data/train.parquet"
VALIDATION_PATH = "outputs/val_structs.csv"
OUTPUT_PATH = "outputs/clean_train.parquet"

BATCH_SIZE = 50000
MASS_TOLERANCE_PPM = 10

COLUMNS = [
    "ingest_lib",
    "normalized_smiles",
    "inchikey14",
    "molecular_formula",
    "ionization_mode",
    "instrument_type",
    "adduct",
    "precursor_mz",
    "ms2_mzs",
    "ms2_normalized_intensities",
    "num_peaks",
    "base_peak_intensity",
    "collision_energy_ev",
]


# --------------------------------------------------
# Load validation structures
# --------------------------------------------------

print("Loading validation structures...")

validation = set(
    pd.read_csv(VALIDATION_PATH)["inchikey14"].dropna()
)

print("Held-out structures:", len(validation))


# --------------------------------------------------
# Prepare output
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)

if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)

source = pq.ParquetFile(TRAIN_PATH)
output_schema = pa.schema([
    source.schema_arrow.field(name)
    for name in COLUMNS
])
writer = None

total = 0
kept = 0
excluded_validation = 0
excluded_mass = 0
excluded_invalid = 0

# Cache exact molecular masses.
mass_cache = {}


# --------------------------------------------------
# Process training data in batches
# --------------------------------------------------

print("\nPreparing training data...")

for batch_number, batch in enumerate(
    source.iter_batches(
        batch_size=BATCH_SIZE,
        columns=COLUMNS
    ),
    start=1
):
    df = batch.to_pandas()
    total += len(df)

    # Exclude validation structures.
    validation_mask = df["inchikey14"].isin(validation)

    excluded_validation += int(validation_mask.sum())

    df = df.loc[~validation_mask].copy()

    if df.empty:
        continue

    keep_mask = np.zeros(len(df), dtype=bool)

    # Check each remaining spectrum once.
    for position, (mz, adduct, smiles) in enumerate(
        zip(
            df["precursor_mz"],
            df["adduct"],
            df["normalized_smiles"]
        )
    ):

        if adduct not in ADDUCTS or pd.isna(mz) or not isinstance(smiles, str) or not smiles:
            excluded_invalid += 1
            continue

        # Calculate molecular mass only once per unique SMILES.
        if smiles not in mass_cache:
            mol = Chem.MolFromSmiles(smiles)

            mass_cache[smiles] = (
                Descriptors.ExactMolWt(mol)
                if mol is not None else None
            )

        exact_mass = mass_cache[smiles]

        if exact_mass is None:
            excluded_invalid += 1
            continue

        try:
            calculated_mass = neutral_mass_from_mz(
                float(mz),
                adduct
            )

            error = abs(
                ppm_error(calculated_mass, exact_mass)
            )

        except (ValueError, TypeError):
            excluded_invalid += 1
            continue

        if error <= MASS_TOLERANCE_PPM:
            keep_mask[position] = True
        else:
            excluded_mass += 1

    clean = df.loc[keep_mask]

    # Write accepted records.
    if not clean.empty:

        table = pa.Table.from_pandas(
    clean,
    schema=output_schema,
    preserve_index=False,
    safe=False
)

        if writer is None:
            writer = pq.ParquetWriter(
                OUTPUT_PATH,
                table.schema,
                compression="zstd"
            )

        writer.write_table(table)

        kept += len(clean)

    print(
        f"Batch {batch_number}: "
        f"processed={total:,} "
        f"kept={kept:,}"
    )


# --------------------------------------------------
# Finish
# --------------------------------------------------

if writer is not None:
    writer.close()

print("\nPreparation finished.")

print("Total processed:", f"{total:,}")
print("Kept:", f"{kept:,}")
print("Excluded validation:", f"{excluded_validation:,}")
print("Excluded mass mismatch:", f"{excluded_mass:,}")
print("Excluded invalid:", f"{excluded_invalid:,}")

if writer is not None:
    print("Output:", OUTPUT_PATH)
else:
    print("WARNING: No valid training records were written.")
