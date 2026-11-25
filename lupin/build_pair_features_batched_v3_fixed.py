
import argparse
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import extract_features
from candidate_features_v2 import extract_molecular_features_v2

from build_pair_features import read_pair_sample, build_spectrum_features


CONFIG = {
    "train": {
        "spectra": "outputs/ranking_train.parquet",
        "catalog": "outputs/train_candidate_catalog.parquet",
        "pairs": "outputs/training_pairs_v3_fixed.parquet",
              "output": "outputs/train_pair_features_v3_fixed.parquet",
    },
    "valid": {
        "spectra": "outputs/ranking_valid.parquet",
        "catalog": "outputs/validation_candidate_catalog.parquet",
        "pairs": "outputs/validation_pairs.parquet",
        "output": "outputs/valid_pair_features_v2_full.parquet",
    },
}

SPECTRUM_BATCH_SIZE = 10000
CATALOG_BATCH_SIZE = 10000
PAIR_BATCH_SIZE = 25000


def build_spectrum_cache(path, cache_path):
    source = pq.ParquetFile(path)
    total = source.metadata.num_rows

    columns = [
        "precursor_mz",
        "adduct",
        "ms2_mzs",
        "ms2_normalized_intensities",
        "collision_energy_ev",
        "ionization_mode",
        "instrument_type",
    ]

    cache = None
    names = None
    offset = 0

    print("\nBuilding spectrum feature cache...")
    print("Total spectra:", f"{total:,}")

    for batch in source.iter_batches(
        batch_size=SPECTRUM_BATCH_SIZE,
        columns=columns,
    ):
        data = batch.to_pydict()

        for i in range(batch.num_rows):
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

                if not names:
                    raise ValueError("No spectrum features generated")

                cache = np.lib.format.open_memmap(
                    cache_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(total, len(names)),
                )

            cache[offset + i] = [result[name] for name in names]

        offset += batch.num_rows

        if offset % 100000 < SPECTRUM_BATCH_SIZE:
            print(f"  Spectra processed: {offset:,}/{total:,}")

    if offset != total or cache is None:
        raise ValueError("Spectrum cache is incomplete")

    cache.flush()
    del cache

    print("Spectrum features:", len(names))
    print("Spectrum cache saved:", cache_path)

    return names


def build_candidate_cache(path, cache_path):
    source = pq.ParquetFile(path)
    total = source.metadata.num_rows

    cache = None
    names = None
    candidate_index = {}
    offset = 0

    print("\nBuilding V3 candidate feature cache...")
    print("Total candidates:", f"{total:,}")

    for batch in source.iter_batches(
        batch_size=CATALOG_BATCH_SIZE,
        columns=["inchikey14", "smiles"],
    ):
        data = batch.to_pydict()

        for i, (key, smiles) in enumerate(
            zip(data["inchikey14"], data["smiles"])
        ):
            if key in candidate_index:
                raise ValueError(f"Duplicate candidate key: {key}")

            result = extract_molecular_features_v2(smiles)

            if names is None:
                names = list(result.keys())

                if len(names) != 30:
                    raise ValueError(
                        f"Expected 30 candidate features, got {len(names)}"
                    )

                cache = np.lib.format.open_memmap(
                    cache_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(total, len(names)),
                )

            row_index = offset + i
            candidate_index[key] = row_index
            cache[row_index] = [result[name] for name in names]

        offset += batch.num_rows

        if offset % 100000 < CATALOG_BATCH_SIZE:
            print(f"  Candidates processed: {offset:,}/{total:,}")

    if offset != total or cache is None:
        raise ValueError("Candidate cache is incomplete")

    cache.flush()
    del cache

    print("Candidate features:", len(names))
    print("Candidate cache saved:", cache_path)
    print("Candidate index entries:", len(candidate_index))

    return names, candidate_index


def make_feature_schema(spectrum_names, candidate_names):
    feature_names = (
        spectrum_names
        + candidate_names
        + [
            "candidate_mass_error_ppm",
            "candidate_abs_mass_error_ppm",
        ]
    )

    expected_count = len(spectrum_names) + 32

    if len(candidate_names) != 30:
        raise ValueError(
            f"Expected 30 candidate features, got {len(candidate_names)}"
        )

    if len(feature_names) != expected_count:
        raise ValueError(
            f"Expected {expected_count} total features, got {len(feature_names)}"
        )

    return feature_names


def calculate_mass_errors(
    spectrum_values,
    candidate_values,
    spectrum_names,
    candidate_names,
):
    neutral_mass = spectrum_values[
        :, spectrum_names.index("neutral_mass")
    ]

    exact_mass = candidate_values[
        :, candidate_names.index("candidate_exact_mass")
    ]

    mass_error = np.full(
        len(exact_mass),
        999999.0,
        dtype=np.float32,
    )

    valid_mass = exact_mass > 0

    mass_error[valid_mass] = (
        (
            neutral_mass[valid_mass] - exact_mass[valid_mass]
        )
        / exact_mass[valid_mass]
        * 1_000_000
    )

    return mass_error


def write_full_pair_features(
    config,
    spectrum_names,
    candidate_names,
    spectrum_cache_path,
    candidate_cache_path,
    candidate_index,
):
    spectra = np.load(spectrum_cache_path, mmap_mode="r")
    candidates = np.load(candidate_cache_path, mmap_mode="r")

    feature_names = make_feature_schema(
        spectrum_names,
        candidate_names,
    )

    schema = pa.schema(
        [
            pa.field("spectrum_idx", pa.int64()),
            pa.field("candidate_inchikey14", pa.string()),
            pa.field("label", pa.int8()),
        ] + [
            pa.field(name, pa.float32())
            for name in feature_names
        ]
    )

    os.makedirs(os.path.dirname(config["output"]), exist_ok=True)

    if os.path.exists(config["output"]):
        raise FileExistsError(
            f"Refusing to overwrite existing V3 output: {config['output']}"
        )

    pair_file = pq.ParquetFile(config["pairs"])
    total = pair_file.metadata.num_rows
    processed = 0

    print("\nGenerating V3 pair features...")
    print("Total pairs:", f"{total:,}")
    print("Total features:", len(feature_names))

    writer = pq.ParquetWriter(
        config["output"],
        schema,
        compression="zstd",
    )

    try:
        for batch in pair_file.iter_batches(
            batch_size=PAIR_BATCH_SIZE,
            columns=[
                "spectrum_idx",
                "candidate_inchikey14",
                "label",
            ],
        ):
            data = batch.to_pydict()

            spectrum_ids = np.asarray(
                data["spectrum_idx"],
                dtype=np.int64,
            )

            candidate_keys = data["candidate_inchikey14"]

            try:
                candidate_ids = np.fromiter(
                    (candidate_index[key] for key in candidate_keys),
                    dtype=np.int64,
                    count=len(candidate_keys),
                )
            except KeyError as exc:
                raise ValueError(
                    f"Candidate missing from catalog: {exc}"
                )

            if (
                np.any(spectrum_ids < 0)
                or np.any(spectrum_ids >= spectra.shape[0])
            ):
                raise ValueError("Invalid spectrum index")

            spectrum_values = np.asarray(
                spectra[spectrum_ids],
                dtype=np.float32,
            )

            candidate_values = np.asarray(
                candidates[candidate_ids],
                dtype=np.float32,
            )

            mass_error = calculate_mass_errors(
                spectrum_values,
                candidate_values,
                spectrum_names,
                candidate_names,
            )

            features = np.column_stack(
                [
                    spectrum_values,
                    candidate_values,
                    mass_error,
                    np.abs(mass_error),
                ]
            ).astype(np.float32, copy=False)

            if not np.isfinite(features).all():
                raise ValueError(
                    f"Non-finite features near pair {processed}"
                )

            arrays = [
                pa.array(spectrum_ids, type=pa.int64()),
                pa.array(candidate_keys, type=pa.string()),
                pa.array(data["label"], type=pa.int8()),
            ]

            arrays.extend(
                pa.array(features[:, i], type=pa.float32())
                for i in range(features.shape[1])
            )

            output_table = pa.Table.from_arrays(
                arrays,
                schema=schema,
            )

            writer.write_table(output_table)
            processed += batch.num_rows

            print(f"  Pairs processed: {processed:,}/{total:,}")

    finally:
        writer.close()

    if processed != total:
        raise ValueError(
            f"Expected {total} pairs, processed {processed}"
        )

    print("\nV3 feature file saved:", config["output"])
    print("Rows:", f"{processed:,}")
    print("Features:", len(feature_names))


def run_small_test(config, split, max_pairs):
    print("Mode: small V3 selective test")

    pair_table, total = read_pair_sample(
        config["pairs"],
        max_pairs,
    )

    print("Total available pairs:", f"{total:,}")
    print("Pairs selected:", f"{pair_table.num_rows:,}")

    pair_data = pair_table.to_pydict()

    required_spectra = set(pair_data["spectrum_idx"])
    required_candidates = set(pair_data["candidate_inchikey14"])

    spectrum_features, spectrum_names = build_spectrum_features(
        config["spectra"],
        required_spectra,
    )

    print("Spectrum features:", len(spectrum_names))

    if not spectrum_names:
        raise ValueError("No spectrum features generated")

    catalog = pq.ParquetFile(config["catalog"])
    candidate_features = {}
    candidate_names = None

    for batch in catalog.iter_batches(
        batch_size=CATALOG_BATCH_SIZE,
        columns=["inchikey14", "smiles"],
    ):
        data = batch.to_pydict()

        for key, smiles in zip(
            data["inchikey14"],
            data["smiles"],
        ):
            if key not in required_candidates:
                continue

            result = extract_molecular_features_v2(smiles)

            if candidate_names is None:
                candidate_names = list(result.keys())

            candidate_features[key] = result

    if set(candidate_features) != required_candidates:
        missing = required_candidates - set(candidate_features)
        raise ValueError(
            f"Missing candidate features: {len(missing)}"
        )

    if candidate_names is None or len(candidate_names) != 30:
        raise ValueError("Expected exactly 30 V2 candidate features")

    feature_names = make_feature_schema(
        spectrum_names,
        candidate_names,
    )

    spectrum_ids = np.asarray(
        pair_data["spectrum_idx"],
        dtype=np.int64,
    )

    candidate_keys = pair_data["candidate_inchikey14"]

    spectrum_values = np.asarray(
        [
            [spectrum_features[int(i)][name] for name in spectrum_names]
            for i in spectrum_ids
        ],
        dtype=np.float32,
    )

    candidate_values = np.asarray(
        [
            [candidate_features[key][name] for name in candidate_names]
            for key in candidate_keys
        ],
        dtype=np.float32,
    )

    mass_error = calculate_mass_errors(
        spectrum_values,
        candidate_values,
        spectrum_names,
        candidate_names,
    )

    features = np.column_stack(
        [
            spectrum_values,
            candidate_values,
            mass_error,
            np.abs(mass_error),
        ]
    ).astype(np.float32, copy=False)

    if not np.isfinite(features).all():
        raise ValueError("Non-finite values in V3 test features")

    arrays = [
        pa.array(spectrum_ids, type=pa.int64()),
        pa.array(candidate_keys, type=pa.string()),
        pa.array(pair_data["label"], type=pa.int8()),
    ]

    arrays.extend(
        pa.array(features[:, i], type=pa.float32())
        for i in range(features.shape[1])
    )

    output_path = (
        f"outputs/{split}_pair_features_v3_test.parquet"
    )

    if os.path.exists(output_path):
        raise FileExistsError(
            f"Refusing to overwrite existing test file: {output_path}"
        )

    table = pa.Table.from_arrays(
        arrays,
        names=[
            "spectrum_idx",
            "candidate_inchikey14",
            "label",
        ] + feature_names,
    )

    pq.write_table(table, output_path, compression="zstd")

    print("\nV3 small test completed.")
    print("Output:", output_path)
    print("Rows:", table.num_rows)
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

    os.makedirs("outputs", exist_ok=True)

    print("Split:", args.split)

    if args.max_pairs is not None:
        run_small_test(
            config,
            args.split,
            args.max_pairs,
        )
        return

    print("Mode: full V3 batched processing")

    spectrum_cache_path = (
        f"outputs/spectrum_features_v3_fixed_{args.split}.npy"
    )

    candidate_cache_path = (
        f"outputs/candidate_features_v3_fixed_{args.split}.npy"
    )

    for path in [spectrum_cache_path, candidate_cache_path]:
        if os.path.exists(path):
            raise FileExistsError(
                f"Refusing to overwrite existing V3 cache: {path}"
           )

    spectrum_names = build_spectrum_cache(
        config["spectra"],
        spectrum_cache_path,
    )

    candidate_names, candidate_index = build_candidate_cache(
        config["catalog"],
        candidate_cache_path,
    )

    write_full_pair_features(
        config=config,
        spectrum_names=spectrum_names,
        candidate_names=candidate_names,
        spectrum_cache_path=spectrum_cache_path,
        candidate_cache_path=candidate_cache_path,
        candidate_index=candidate_index,
    )

    print("\nFull V3 feature generation completed.")


if __name__ == "__main__":
    main()
