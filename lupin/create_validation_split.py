
import pyarrow.parquet as pq
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

INPUT_PATH = "outputs/clean_train.parquet"
OUTPUT_TRAIN = "outputs/ranking_train.parquet"
OUTPUT_VALID = "outputs/ranking_valid.parquet"

VALID_FRACTION = 0.20
SEED = 42

print("Reading unique molecule structures...")

parquet = pq.ParquetFile(INPUT_PATH)

all_keys = set()

for batch in parquet.iter_batches(
    batch_size=50000,
    columns=["inchikey14"],
):
    for key in batch.column("inchikey14").to_pylist():
        if key:
            all_keys.add(key)

keys = sorted(all_keys)

rng = np.random.default_rng(SEED)
rng.shuffle(keys)

valid_count = int(len(keys) * VALID_FRACTION)

valid_keys = set(keys[:valid_count])
train_keys = set(keys[valid_count:])

print(f"Unique structures: {len(keys):,}")
print(f"Training structures: {len(train_keys):,}")
print(f"Validation structures: {len(valid_keys):,}")

train_writer = None
valid_writer = None

print("\nWriting split parquet files...")

for batch in parquet.iter_batches(batch_size=50000):
    table = pa.Table.from_batches([batch])

    key_column = table.column("inchikey14").to_pylist()

    train_mask = [key in train_keys for key in key_column]
    valid_mask = [key in valid_keys for key in key_column]

    train_table = table.filter(pa.array(train_mask))
    valid_table = table.filter(pa.array(valid_mask))

    if train_writer is None:
        train_writer = pq.ParquetWriter(
            OUTPUT_TRAIN,
            train_table.schema
        )
        valid_writer = pq.ParquetWriter(
            OUTPUT_VALID,
            valid_table.schema
        )

    if train_table.num_rows:
        train_writer.write_table(train_table)

    if valid_table.num_rows:
        valid_writer.write_table(valid_table)

train_writer.close()
valid_writer.close()

print("\nValidation split completed.")
print(f"Training file: {OUTPUT_TRAIN}")
print(f"Validation file: {OUTPUT_VALID}")
