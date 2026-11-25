import argparse
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import extract_features
from candidate_features import extract_molecular_features


CONFIG = {
    "train": {
        "spectra": "outputs/ranking_train.parquet",
        "catalog": "outputs/train_candidate_catalog.parquet",
        "pairs": "outputs/training_pairs.parquet",
        "output": "outputs/train_pair_features.parquet",
    },
    "valid": {
        "spectra": "outputs/ranking_valid.parquet",
        "catalog": "outputs/validation_candidate_catalog.parquet",
        "pairs": "outputs/validation_pairs.parquet",
        "output": "outputs/validation_pair_features.parquet",
    },
}

PAIR_BATCH_SIZE = 25000
SPECTRUM_BATCH_SIZE = 10000
CATALOG_BATCH_SIZE = 10000


def read_pair_sample(path, max_pairs=None):
    """Read only the requested number of pairs."""

    parquet = pq.ParquetFile(path)
    total = parquet.metadata.num_rows

    limit = total if max_pairs is None else min(total, max_pairs)

    parts = []
    remaining = limit

    for batch in parquet.iter_batches(
        batch_size=PAIR_BATCH_SIZE,
        columns=[
            "spectrum_idx",
            "candidate_inchikey14",
            "label",
        ],
    ):
        take = min(remaining, batch.num_rows)
        parts.append(batch.slice(0, take))
        remaining -= take

        if remaining == 0:
            break

    table = pa.Table.from_batches(parts)

    if table.num_rows != limit:
        raise ValueError("Could not read requested pair count")

    return table, total


def build_spectrum_features(path, required_indices):
    """Calculate features only for required spectrum rows."""

    required = set(int(x) for x in required_indices)

    if not required:
        raise ValueError("No spectrum indices requested")

    max_index = max(required)

    source = pq.ParquetFile(path)

    columns = [
        "precursor_mz",
        "adduct",
        "ms2_mzs",
        "ms2_normalized_intensities",
        "collision_energy_ev",
        "ionization_mode",
        "instrument_type",
    ]

    results = {}
    row_offset = 0
    names = None

    print("\nExtracting required spectrum features...")

    for batch in source.iter_batches(
        batch_size=SPECTRUM_BATCH_SIZE,
        columns=columns,
    ):

        data = batch.to_pydict()

        for i in range(batch.num_rows):

            idx = row_offset + i

            if idx in required:

                result = extract_features(
                    precursor_mz=data["precursor_mz"][i],
                    adduct=data["adduct"][i],
                    mz_values=data["ms2_mzs"][i],
                    intensities=data["ms2_normalized_intensities"][i],
                    collision_energy=data["collision_energy_ev"][i],
                    ionization_mode=data["ionization_mode"][i],
                    instrument_type=data["instrument_type"][i],
                )

                if names is None:
                    names = list(result.keys())

                    if len(names) != 38:
                        raise ValueError(
                            f"Expected 38 spectrum features, got {len(names)}"
                        )

                results[idx] = result

        row_offset += batch.num_rows

        if row_offset > max_index and len(results) == len(required):
            break

        if row_offset % 100000 < SPECTRUM_BATCH_SIZE:
            print(
                f"  Scanned {row_offset:,} spectra; "
                f"found {len(results):,}/{len(required):,}"
            )

    if len(results) != len(required):
        missing = required - set(results)
        raise ValueError(
            f"Missing spectrum indices: {list(missing)[:10]}"
        )

    print("Spectrum features:", len(results))

    return results, names


def build_candidate_features(path, required_keys):
    """Calculate descriptors only for candidates used by the pairs."""

    required = set(required_keys)

    if not required:
        raise ValueError("No candidate keys requested")

    source = pq.ParquetFile(path)

    results = {}
    names = None

    print("\nExtracting required candidate features...")

    for batch in source.iter_batches(
        batch_size=CATALOG_BATCH_SIZE,
        columns=["inchikey14", "smiles"],
    ):

        data = batch.to_pydict()

        for key, smiles in zip(
            data["inchikey14"],
            data["smiles"],
        ):

            if key not in required:
                continue

            if key in results:
                raise ValueError(f"Duplicate candidate key: {key}")

            features = extract_molecular_features(smiles)

            if names is None:
                names = list(features.keys())

                if len(names) != 18:
                    raise ValueError(
                        f"Expected 18 candidate features, got {len(names)}"
                    )

            results[key] = features

        if len(results) == len(required):
            break

    if len(results) != len(required):
        missing = required - set(results)
        raise ValueError(
            f"Missing candidate keys: {list(missing)[:10]}"
        )

    print("Candidate features:", len(results))

    return results, names


def generate_features(
    pair_table,
    spectrum_features,
    candidate_features,
    spectrum_names,
    candidate_names,
    output_path,
):
    """Combine spectrum and candidate features and write Parquet."""

    feature_names = (
        spectrum_names
        + candidate_names
        + [
            "candidate_mass_error_ppm",
            "candidate_abs_mass_error_ppm",
        ]
    )

    if len(feature_names) != 58:
        raise ValueError(
            f"Expected 58 features, got {len(feature_names)}"
        )

    schema = pa.schema(
        [
            pa.field("spectrum_idx", pa.int64()),
            pa.field("candidate_inchikey14", pa.string()),
            pa.field("label", pa.int8()),
        ]
        + [
            pa.field(name, pa.float32())
            for name in feature_names
        ]
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        os.remove(output_path)

    writer = pq.ParquetWriter(
        output_path,
        schema,
        compression="zstd",
    )

    try:
        data = pair_table.to_pydict()

        spectrum_ids = data["spectrum_idx"]
        candidate_keys = data["candidate_inchikey14"]
        labels = data["label"]

        feature_arrays = {
            name: np.empty(len(labels), dtype=np.float32)
            for name in feature_names
        }

        for i, (spectrum_idx, candidate_key) in enumerate(
            zip(spectrum_ids, candidate_keys)
        ):

            spectrum = spectrum_features[spectrum_idx]
            candidate = candidate_features[candidate_key]

            exact_mass = candidate["candidate_exact_mass"]
            neutral_mass = spectrum["neutral_mass"]

            mass_error = (
                (neutral_mass - exact_mass)
                / exact_mass
                * 1_000_000
                if exact_mass > 0 else 999999.0
            )

            for name in spectrum_names:
                feature_arrays[name][i] = spectrum[name]

            for name in candidate_names:
                feature_arrays[name][i] = candidate[name]

            feature_arrays["candidate_mass_error_ppm"][i] = mass_error
            feature_arrays["candidate_abs_mass_error_ppm"][i] = abs(
                mass_error
            )

        output_data = {
            "spectrum_idx": spectrum_ids,
            "candidate_inchikey14": candidate_keys,
            "label": labels,
        }

        output_data.update(feature_arrays)

        table = pa.Table.from_pydict(
            output_data,
            schema=schema,
        )

        if not np.isfinite(
            np.column_stack(
                [feature_arrays[name] for name in feature_names]
            )
        ).all():
            raise ValueError("Non-finite feature values detected")

        writer.write_table(table)

    finally:
        writer.close()

    print("\nFeature file saved:", output_path)
    print("Rows:", pair_table.num_rows)
    print("Features:", len(feature_names))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["train", "valid"],
        required=True,
    )

    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if args.max_pairs is not None and args.max_pairs <= 0:
        parser.error("--max-pairs must be positive")

    config = CONFIG[args.split]

    print("Split:", args.split)

    # Read the requested pair rows first.
    pair_table, total_pairs = read_pair_sample(
        config["pairs"],
        args.max_pairs,
    )

    print("Total available pairs:", f"{total_pairs:,}")
    print("Pairs selected:", f"{pair_table.num_rows:,}")

    pair_data = pair_table.to_pydict()

    required_spectra = set(pair_data["spectrum_idx"])
    required_candidates = set(pair_data["candidate_inchikey14"])

    print("Unique spectra required:", len(required_spectra))
    print("Unique candidates required:", len(required_candidates))

    # Extract only the required features.
    spectrum_features, spectrum_names = build_spectrum_features(
        config["spectra"],
        required_spectra,
    )

    candidate_features, candidate_names = build_candidate_features(
        config["catalog"],
        required_candidates,
    )

    generate_features(
        pair_table=pair_table,
        spectrum_features=spectrum_features,
        candidate_features=candidate_features,
        spectrum_names=spectrum_names,
        candidate_names=candidate_names,
        output_path=config["output"],
    )

    print("\nCompleted successfully.")


if __name__ == "__main__":
    main()
