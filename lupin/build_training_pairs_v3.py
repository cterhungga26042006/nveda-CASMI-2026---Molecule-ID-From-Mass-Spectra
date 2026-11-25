import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rdkit import Chem
from rdkit.Chem import Descriptors

from adducts import neutral_mass_from_mz


SPECTRA_PATH = "outputs/ranking_train.parquet"
CATALOG_PATH = "outputs/train_candidate_catalog.parquet"

SEED = 42
MAX_NEGATIVES = 32
FORMULA_NEGATIVES = 8
NEAREST_NEGATIVES = 8
BATCH_SIZE = 5000
TOLERANCE_PPM = 10.0


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--max-spectra", type=int, default=None)
args = parser.parse_args()

OUTPUT_PATH = Path(args.output)

if OUTPUT_PATH.exists():
    raise FileExistsError(
        f"Refusing to overwrite existing file: {OUTPUT_PATH}"
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

print("Loading candidate catalog...")

catalog = pq.read_table(
    CATALOG_PATH,
    columns=["inchikey14", "smiles", "molecular_formula"],
)

keys = catalog["inchikey14"].to_pylist()
smiles_list = catalog["smiles"].to_pylist()
formulas = catalog["molecular_formula"].to_pylist()

masses = np.empty(len(keys), dtype=np.float64)
valid = np.ones(len(keys), dtype=bool)

for i, smiles in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        valid[i] = False
        masses[i] = np.nan
    else:
        masses[i] = Descriptors.ExactMolWt(mol)

if not valid.all():
    keys = [k for k, ok in zip(keys, valid) if ok]
    formulas = [f for f, ok in zip(formulas, valid) if ok]
    masses = masses[valid]

order = np.argsort(masses)

masses = masses[order]
keys = [keys[i] for i in order]
formulas = [formulas[i] for i in order]

print("Valid catalog candidates:", len(keys))

formula_candidates = defaultdict(list)

for i, formula in enumerate(formulas):
    formula_candidates[formula].append(i)

spectra_file = pq.ParquetFile(SPECTRA_PATH)

schema = pa.schema([
    ("spectrum_idx", pa.int64()),
    ("candidate_inchikey14", pa.string()),
    ("label", pa.int8()),
])

writer = pq.ParquetWriter(
    OUTPUT_PATH,
    schema,
    compression="zstd",
)

total_spectra = 0
processed_spectra = 0
positive_found = 0
positive_missing = 0
invalid_spectra = 0
total_pairs = 0
negative_counts = []

batch_spectrum_indices = []
batch_candidate_keys = []
batch_labels = []

ppm_fraction = TOLERANCE_PPM / 1_000_000


def flush():
    global total_pairs

    if not batch_spectrum_indices:
        return

    table = pa.Table.from_pydict(
        {
            "spectrum_idx": batch_spectrum_indices,
            "candidate_inchikey14": batch_candidate_keys,
            "label": batch_labels,
        },
        schema=schema,
    )

    writer.write_table(table)
    total_pairs += len(batch_spectrum_indices)

    batch_spectrum_indices.clear()
    batch_candidate_keys.clear()
    batch_labels.clear()


print("Generating V3 training pairs...")

stop = False

for batch in spectra_file.iter_batches(
    batch_size=BATCH_SIZE,
    columns=[
        "precursor_mz",
        "adduct",
        "inchikey14",
        "molecular_formula",
    ],
):
    data = batch.to_pydict()

    for j in range(batch.num_rows):
        if (
            args.max_spectra is not None
            and processed_spectra >= args.max_spectra
        ):
            stop = True
            break

        spectrum_idx = total_spectra
        total_spectra += 1
        processed_spectra += 1

        try:
            neutral_mass = neutral_mass_from_mz(
                data["precursor_mz"][j],
                data["adduct"][j],
            )
        except Exception:
            invalid_spectra += 1
            continue

        if neutral_mass is None or not np.isfinite(neutral_mass):
            invalid_spectra += 1
            continue

        lower = neutral_mass / (1 + ppm_fraction)
        upper = neutral_mass / (1 - ppm_fraction)

        left = np.searchsorted(masses, lower, side="left")
        right = np.searchsorted(masses, upper, side="right")

        if left >= right:
            positive_missing += 1
            continue

        true_key = data["inchikey14"][j]
        true_formula = data["molecular_formula"][j]

        pool = np.arange(left, right, dtype=np.int32)

        positive_positions = np.flatnonzero(
            np.array([keys[k] == true_key for k in pool])
        )

        if len(positive_positions) == 0:
            positive_missing += 1
            continue

        positive_found += 1
        positive_idx = int(pool[positive_positions[0]])

        negative_pool = pool[pool != positive_idx]

        if len(negative_pool) == 0:
            batch_spectrum_indices.append(spectrum_idx)
            batch_candidate_keys.append(true_key)
            batch_labels.append(1)
            negative_counts.append(0)

            if len(batch_spectrum_indices) >= 300000:
                flush()

            continue

        selected = []
        selected_set = set()
        def add_candidates(candidates, limit):
            available = [
                int(x) for x in candidates
                if int(x) != positive_idx
                and int(x) not in selected_set
            ]

            if not available or limit <= 0:
                return

            count = min(limit, len(available))

            if len(available) > count:
                chosen = rng.choice(
                    available,
                    size=count,
                    replace=False,
                ).tolist()
            else:
                chosen = available

            for candidate_idx in chosen:
                if candidate_idx not in selected_set:
                    selected.append(candidate_idx)
                    selected_set.add(candidate_idx)

        # 1. Same-formula hard negatives.
        formula_pool = formula_candidates.get(true_formula, [])
        formula_pool = [
            x for x in formula_pool
            if left <= x < right and x != positive_idx
        ]

        add_candidates(formula_pool, FORMULA_NEGATIVES)

        # 2. Candidates with the closest precursor mass.
        mass_errors = (
            (neutral_mass - masses[negative_pool])
            / masses[negative_pool]
        ) * 1_000_000

        nearest_order = np.argsort(np.abs(mass_errors))
        nearest_pool = negative_pool[nearest_order]

        add_candidates(nearest_pool, NEAREST_NEGATIVES)

        # 3. Fill remaining slots with random mass-matched negatives.
        remaining_pool = [
            int(x) for x in negative_pool
            if int(x) not in selected_set
        ]

        remaining_slots = MAX_NEGATIVES - len(selected)

        add_candidates(remaining_pool, remaining_slots)

        # Write the positive.
        batch_spectrum_indices.append(spectrum_idx)
        batch_candidate_keys.append(true_key)
        batch_labels.append(1)

        # Write selected negatives.
        for candidate_idx in selected:
            batch_spectrum_indices.append(spectrum_idx)
            batch_candidate_keys.append(keys[candidate_idx])
            batch_labels.append(0)

        negative_counts.append(len(selected))

        if len(batch_spectrum_indices) >= 300000:
            flush()

    if processed_spectra % 50000 < BATCH_SIZE:
        print(
            f"Processed: {processed_spectra:,} | "
            f"Positive found: {positive_found:,} | "
            f"Positive missing: {positive_missing:,}"
        )

    if stop:
        break

flush()
writer.close()

print("\nV3 PAIR GENERATION COMPLETE")
print("Output:", OUTPUT_PATH)
print("Spectra processed:", f"{processed_spectra:,}")
print("Positive found:", f"{positive_found:,}")
print("Positive missing:", f"{positive_missing:,}")
print("Invalid spectra:", f"{invalid_spectra:,}")
print("Total pairs:", f"{total_pairs:,}")

if negative_counts:
    counts = np.array(negative_counts)

    print("Average negatives:", round(float(counts.mean()), 2))
    print("Median negatives:", float(np.median(counts)))
    print("Minimum negatives:", int(counts.min()))
    print("Maximum negatives:", int(counts.max()))

print("File size (MB):", round(OUTPUT_PATH.stat().st_size / 1_000_000, 2))
