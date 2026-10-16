import argparse
import math

import pandas as pd
import pyarrow.parquet as pq

# These columns hold lists, which CSV can't store, so they become "a;b;c" text.
LIST_COLS = ["ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev"]
MAX_FULL_ROWS = 100_000  # refuse to convert huge files in full


def lists_to_text(df):
    for c in LIST_COLS:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda x: ";".join(map(str, x)) if x is not None else ""
            )
    return df


def main():
    p = argparse.ArgumentParser(description="Convert a parquet file to CSV.")
    p.add_argument("input", help="path to the .parquet file")
    p.add_argument("-o", "--output", help="output CSV path (default: same name .csv)")
    p.add_argument("-n", "--rows", type=int, default=None,
                   help="random sample of N rows (needed for big files)")
    p.add_argument("--seed", type=int, default=0, help="random seed for the sample")
    a = p.parse_args()

    out = a.output or a.input.rsplit(".", 1)[0] + ".csv"
    pf = pq.ParquetFile(a.input)
    total = pf.metadata.num_rows
    print(f"{a.input}: {total:,} rows, {pf.num_row_groups} row groups")

    if a.rows is None:
        if total > MAX_FULL_ROWS:
            raise SystemExit(
                f"{total:,} rows is too many for one CSV. Use -n, for example: -n 1000"
            )
        df = pf.read().to_pandas()
    else:
        # take an equal random share from every row group, one group at a time
        per_group = math.ceil(a.rows / pf.num_row_groups)
        parts = []
        for i in range(pf.num_row_groups):
            g = pf.read_row_group(i).to_pandas()
            parts.append(g.sample(n=min(per_group, len(g)), random_state=a.seed))
        df = pd.concat(parts).head(a.rows)

    df = lists_to_text(df)
    df.to_csv(out, index=False)
    print(f"saved {df.shape[0]:,} rows x {df.shape[1]} columns -> {out}")


if __name__ == "__main__":
    main()
