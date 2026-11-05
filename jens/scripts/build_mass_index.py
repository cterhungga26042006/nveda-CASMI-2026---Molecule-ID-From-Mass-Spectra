from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DB_PATH = ROOT / "data" / "processed" / "molecule_database.parquet"
OUT_PATH = ROOT / "data" / "processed" / "mass_index.parquet"


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            "Run prepare_database.py first."
        )

    df = pd.read_parquet(DB_PATH)

    df["exact_mass"] = pd.to_numeric(
        df["exact_mass"], errors="coerce"
    )

    df = df.dropna(subset=["exact_mass"])
    df = df.sort_values("exact_mass").reset_index(drop=True)

    df.to_parquet(OUT_PATH, index=False)

    print("Mass index created:", OUT_PATH)
    print("Number of molecules:", len(df))
    print("Minimum mass:", df["exact_mass"].min())
    print("Maximum mass:", df["exact_mass"].max())


if __name__ == "__main__":
    main()