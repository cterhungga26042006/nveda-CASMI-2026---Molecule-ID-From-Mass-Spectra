from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = ROOT / "data" / "raw" / "competition" / "test.csv"
DB_PATH = ROOT / "data" / "processed" / "mass_index.parquet"
OUT_PATH = ROOT / "outputs" / "candidates.csv"
EVIDENCE_PATH = ROOT / "outputs" / "candidate_evidence.csv"

# Example only: singly charged [M+H]+.
# The value is the mass added to neutral M to form the ion.
ADDUCTS = {
    "[M+H]+": 1.0072764666,
}

PPM_TOLERANCE = 10
MAX_CANDIDATES = 100


def mass_matches(database, target_mass, tolerance_ppm):
    """Find neutral molecular masses within a ppm tolerance."""
    if not np.isfinite(target_mass) or target_mass <= 0:
        return database.iloc[0:0]

    delta = target_mass * tolerance_ppm / 1_000_000

    low = target_mass - delta
    high = target_mass + delta

    masses = database["exact_mass"].to_numpy()

    # Find the matching range in a mass-sorted database.
    left = np.searchsorted(masses, low, side="left")
    right = np.searchsorted(masses, high, side="right")

    return database.iloc[left:right]


def main():
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test file: {TEST_PATH}")

    if not DB_PATH.exists():
        raise FileNotFoundError(
            "Run prepare_database.py and build_mass_index.py first."
        )

    test = pd.read_csv(TEST_PATH)
    database = pd.read_parquet(DB_PATH)

    required = {"spectrum_id", "precursor_mz"}

    if not required.issubset(test.columns):
        raise ValueError(
            f"Expected {required}; found {set(test.columns)}. "
            "Update the column names to match your test data."
        )

    test["precursor_mz"] = pd.to_numeric(
        test["precursor_mz"], errors="coerce"
    )

    database["exact_mass"] = pd.to_numeric(
        database["exact_mass"], errors="coerce"
    )

    database = (
        database.dropna(subset=["exact_mass"])
        .sort_values("exact_mass")
        .reset_index(drop=True)
    )

    mass_array = database["exact_mass"].to_numpy()

    candidate_rows = []
    evidence_rows = []

    for _, spectrum in test.iterrows():
        spectrum_id = spectrum["spectrum_id"]
        precursor_mz = spectrum["precursor_mz"]

        if pd.isna(precursor_mz) or precursor_mz <= 0:
            continue

        candidates_for_spectrum = {}

        for adduct, mass_shift in ADDUCTS.items():
            # Valid here only for the configured singly charged adduct.
            neutral_mass = precursor_mz - mass_shift

            matches = mass_matches(
                database,
                neutral_mass,
                PPM_TOLERANCE,
            )

            for _, molecule in matches.iterrows():
                ppm_error = (
                    (molecule["exact_mass"] - neutral_mass)
                    / neutral_mass
                    * 1_000_000
                )

                key = molecule["inchikey14"]

                evidence = {
                    "spectrum_id": spectrum_id,
                    "molecule_id": molecule["molecule_id"],
                    "candidate_smiles": molecule["canonical_smiles"],
                    "mass_error_ppm": float(ppm_error),
                    "adduct_assumption": adduct,
                    "source_database": molecule["source_database"],
                }

                # Keep the best mass error if the same molecule
                # is found more than once.
                old = candidates_for_spectrum.get(key)

                if old is None or abs(ppm_error) < abs(
                    old["mass_error_ppm"]
                ):
                    candidates_for_spectrum[key] = evidence

        ranked = sorted(
            candidates_for_spectrum.values(),
            key=lambda x: abs(x["mass_error_ppm"]),
        )[:MAX_CANDIDATES]

        for item in ranked:
            # Preliminary score only, not a calibrated probability.
            item["score"] = -abs(item["mass_error_ppm"])

            evidence_rows.append(item.copy())

            candidate_rows.append({
                "spectrum_id": item["spectrum_id"],
                "molecule_id": item["molecule_id"],
                "candidate_smiles": item["candidate_smiles"],
                "score": item["score"],
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        candidate_rows,
        columns=[
            "spectrum_id",
            "molecule_id",
            "candidate_smiles",
            "score",
        ],
    ).to_csv(OUT_PATH, index=False)

    pd.DataFrame(evidence_rows).to_csv(
        EVIDENCE_PATH, index=False
    )

    print("Candidate generation complete.")
    print("Spectra processed:", len(test))
    print("Candidate rows:", len(candidate_rows))
    print("Saved:", OUT_PATH)
    print("Evidence:", EVIDENCE_PATH)


if __name__ == "__main__":
    main()