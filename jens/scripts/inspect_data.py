from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/competition")

for path in sorted(DATA_DIR.glob("*")):
    if path.suffix.lower() not in {".csv", ".parquet"}:
        continue

    print(f"\n{'=' * 60}")
    print("File:", path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, nrows=5)
    else:
        df = pd.read_parquet(path).head(5)

    print("Columns:")
    print(df.columns.tolist())
    print("\nFirst rows:")
    print(df.to_string(index=False))