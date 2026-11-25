import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"

FEATURE_FILE = OUTPUT / "train_pair_features_v3_fixed.parquet"
SPECTRA_FILE = OUTPUT / "ranking_train.parquet"

SAMPLE_FILE = OUTPUT / "train_pair_features_v3_fixed_200k.parquet"
MODEL_FILE = OUTPUT / "lupin_ranker_v3_fixed_200k_800.json"
FEATURE_LIST_FILE = OUTPUT / "lupin_ranker_v3_fixed_200k_800_features.json"
IMPORTANCE_FILE = OUTPUT / "lupin_feature_importance_v3_fixed_200k_800.csv"
REPORT_FILE = OUTPUT / "lupin_ranker_v3_fixed_200k_800_report.json"

SAMPLE_GROUPS = 200_000
SEED = 42
READ_BATCH_SIZE = 250_000


def prepare_random_group_sample():
    print("Preparing a random V3 training sample...")
    print("Input:", FEATURE_FILE)

    parquet = pq.ParquetFile(FEATURE_FILE)
    schema_names = parquet.schema_arrow.names

    excluded = {"spectrum_idx", "candidate_inchikey14", "label"}
    feature_cols = [name for name in schema_names if name not in excluded]

    if len(feature_cols) != 70:
        raise ValueError(
            f"Expected 70 features in V3 file, found {len(feature_cols)}"
        )

    required_columns = ["spectrum_idx", "label"] + feature_cols
    missing = [c for c in required_columns if c not in schema_names]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Collecting available spectrum IDs...")

    unique_chunks = []
    last_id = None

    for batch in parquet.iter_batches(
        batch_size=READ_BATCH_SIZE,
        columns=["spectrum_idx"],
    ):
        ids = batch.column(0).to_numpy(zero_copy_only=False)

        if not ids.size:
            continue

        starts = np.flatnonzero(ids[1:] != ids[:-1]) + 1

        if last_id is None or int(ids[0]) != last_id:
            starts = np.r_[0, starts]

        if starts.size:
            unique_chunks.append(ids[starts].astype(np.int64, copy=False))

        last_id = int(ids[-1])

    available_ids = np.concatenate(unique_chunks)
    total_spectra = pq.ParquetFile(SPECTRA_FILE).metadata.num_rows

    if len(available_ids) == 0:
        raise RuntimeError("No available spectrum groups found")

    sample_count = min(SAMPLE_GROUPS, len(available_ids))

    rng = np.random.default_rng(SEED)
    selected_ids = np.sort(
        rng.choice(available_ids, size=sample_count, replace=False)
    )

    print(f"Available spectrum groups: {len(available_ids):,}")
    print(f"Random spectrum IDs selected: {sample_count:,}")
    print(f"Feature count: {len(feature_cols)}")

    writer = None
    total_pairs = 0
    batches_seen = 0
    started = time.perf_counter()

    try:
        for batch in parquet.iter_batches(
            batch_size=READ_BATCH_SIZE,
            columns=required_columns,
        ):
            batches_seen += 1
            ids = batch.column(
                batch.schema.get_field_index("spectrum_idx")
            ).to_numpy(zero_copy_only=False)

            if ids.size:
                if ids.min() < 0 or ids.max() >= total_spectra:
                    raise ValueError(
                        "spectrum_idx values are outside the expected "
                        "0-based ranking_train row range"
                    )

            positions = np.searchsorted(selected_ids, ids)
            mask = positions < selected_ids.size
            valid_positions = np.flatnonzero(mask)
            mask[valid_positions] = (
                selected_ids[positions[valid_positions]]
                == ids[valid_positions]
            )

            if mask.any():
                table = pa.Table.from_batches([batch]).filter(
                    pa.array(mask)
                )

                if writer is None:
                    writer = pq.ParquetWriter(
                        SAMPLE_FILE,
                        table.schema,
                        compression="zstd",
                    )

                writer.write_table(table)
                total_pairs += table.num_rows

            if batches_seen % 20 == 0:
                print(
                    f"  Scanned batches: {batches_seen:,} | "
                    f"sample pairs written: {total_pairs:,}"
                )
    finally:
        if writer is not None:
            writer.close()

    if total_pairs == 0 or not SAMPLE_FILE.exists():
        raise RuntimeError("No sample rows were written")

    print("\nSample file created:", SAMPLE_FILE)
    print(f"Sample pair rows: {total_pairs:,}")
    print(f"Elapsed: {(time.perf_counter() - started) / 60:.1f} minutes")

    return feature_cols, total_pairs


def train_model(feature_cols, expected_pairs):
    print("\nLoading the sampled training data...")
    started = time.perf_counter()

    df = pd.read_parquet(SAMPLE_FILE)

    if len(df) != expected_pairs:
        raise ValueError(
            f"Sample row count changed: expected {expected_pairs:,}, "
            f"loaded {len(df):,}"
        )

    spectrum_ids = df["spectrum_idx"].to_numpy(dtype=np.int64)

    if not np.all(spectrum_ids[1:] >= spectrum_ids[:-1]):
        raise ValueError("Sample spectrum groups are not sorted")

    group_starts = np.r_[
        0,
        np.flatnonzero(spectrum_ids[1:] != spectrum_ids[:-1]) + 1,
    ]
    group_sizes = np.diff(
        np.r_[group_starts, len(spectrum_ids)]
    ).astype(np.uint32)

    y = df["label"].to_numpy(dtype=np.float32)
    positive_counts = np.add.reduceat(y, group_starts)

    if not np.all(positive_counts == 1):
        bad = int(np.sum(positive_counts != 1))
        raise ValueError(
            f"{bad:,} sampled spectrum groups do not have exactly "
            "one positive"
        )

    valid_groups = group_sizes >= 2

    if not np.all(valid_groups):
        excluded_groups = int(np.sum(~valid_groups))
        print(
            f"Excluding {excluded_groups:,} groups "
            "with no negative candidates"
        )

        keep_rows = np.repeat(valid_groups, group_sizes)
        df = df.loc[keep_rows].reset_index(drop=True)

        spectrum_ids = df["spectrum_idx"].to_numpy(dtype=np.int64)

        group_starts = np.r_[
            0,
            np.flatnonzero(spectrum_ids[1:] != spectrum_ids[:-1]) + 1,
        ]

        group_sizes = np.diff(
            np.r_[group_starts, len(spectrum_ids)]
        ).astype(np.uint32)

        y = df["label"].to_numpy(dtype=np.float32)
        positive_counts = np.add.reduceat(y, group_starts)

    if not np.all(positive_counts == 1):
        raise ValueError("A sampled group does not have exactly one positive")

    expected_pairs = len(df)

    print(f"Sample pairs: {len(df):,}")
    print(f"Sample groups: {len(group_sizes):,}")
    print(f"Average candidates per group: {group_sizes.mean():.2f}")
    print(f"Features: {len(feature_cols)}")

    print("\nConverting features to float32...")
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)

    del df, spectrum_ids, group_starts, positive_counts
    gc.collect()

    if not np.isfinite(X).all():
        raise ValueError("Training feature matrix contains NaN or infinity")

    print(f"Feature matrix: {X.shape}, {X.nbytes / 1e9:.2f} GB")
    print("XGBoost version:", xgb.__version__)

    print("\nTraining V3 600-tree XGBoost ranker...")
    model = xgb.XGBRanker(
        objective="rank:pairwise",
        eval_metric="ndcg@25",
        n_estimators=800,
        learning_rate=0.08,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        max_bin=128,
        n_jobs=4,
        random_state=SEED,
    )

    model.fit(
        X,
        y,
        group=group_sizes,
        verbose=25,
    )

    print("\nTraining finished.")
    print(f"Elapsed loading/training: {(time.perf_counter() - started) / 60:.1f} minutes")

    model.save_model(str(MODEL_FILE))

    with open(FEATURE_LIST_FILE, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(IMPORTANCE_FILE, index=False)

    report = {
        "model": MODEL_FILE.name,
        "sample_file": SAMPLE_FILE.name,
        "sample_groups": int(len(group_sizes)),
        "sample_pairs": int(expected_pairs),
        "feature_count": len(feature_cols),
        "seed": SEED,
        "parameters": {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg@25",
            "n_estimators": 800,
            "learning_rate": 0.08,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "max_bin": 128,
            "n_jobs": 4,
            "random_state": SEED,
        },
        "xgboost_version": xgb.__version__,
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSaved V3 600-round model:", MODEL_FILE)
    print("Saved V3 feature list:", FEATURE_LIST_FILE)
    print("Saved V3 feature importance:", IMPORTANCE_FILE)
    print("Saved V3 training report:", REPORT_FILE)

    print("\nTop 20 V3 important features:")
    print(importance.head(20).to_string(index=False))


def main():
    if not SAMPLE_FILE.exists():
        raise FileNotFoundError(SAMPLE_FILE)

    outputs = [
        MODEL_FILE,
        FEATURE_LIST_FILE,
        IMPORTANCE_FILE,
        REPORT_FILE,
    ]

    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing outputs:\n"
            + "\n".join(str(path) for path in existing)
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    parquet = pq.ParquetFile(SAMPLE_FILE)
    schema_names = parquet.schema_arrow.names

    excluded = {"spectrum_idx", "candidate_inchikey14", "label"}
    feature_cols = [
        name for name in schema_names if name not in excluded
    ]

    if len(feature_cols) != 70:
        raise ValueError(
            f"Expected 70 features, found {len(feature_cols)}"
        )

    pair_count = parquet.metadata.num_rows

    print("Reusing existing 200k sample:", SAMPLE_FILE)
    print(f"Sample rows: {pair_count:,}")

    train_model(feature_cols, pair_count)


if __name__ == "__main__":
    main()
