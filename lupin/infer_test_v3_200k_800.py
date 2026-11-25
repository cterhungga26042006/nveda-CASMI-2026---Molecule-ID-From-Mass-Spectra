import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

from rdkit import Chem
from rdkit.Chem import Descriptors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import extract_features
from candidate_features_v2 import extract_molecular_features_v2


OUTPUT = ROOT / "outputs"
TEST_FILE = ROOT / "data" / "test.parquet"
CATALOG_FILE = OUTPUT / "train_candidate_catalog.parquet"

MODEL_FILE = OUTPUT / "lupin_ranker_v3_fixed_200k_800.json"
FEATURE_FILE = OUTPUT / "lupin_ranker_v3_fixed_200k_800_features.json"

RESULT_FILE = OUTPUT / "test_ranked_candidates_lupin_v3_200k_800.csv"

MASS_TOLERANCE_PPM = 10.0
TOP_K = 25


def load_catalog():
    print("\nLoading candidate catalog...")

    table = pq.read_table(
        CATALOG_FILE,
        columns=["inchikey14", "smiles"],
    )

    data = table.to_pydict()

    keys = []
    smiles_list = []
    exact_masses = []

    seen = set()
    invalid = 0

    for key, smiles in zip(data["inchikey14"], data["smiles"]):
        if key is None or smiles is None:
            invalid += 1
            continue

        if key in seen:
            raise ValueError(f"Duplicate catalog key: {key}")

        seen.add(key)

        mol = Chem.MolFromSmiles(str(smiles))

        if mol is None:
            invalid += 1
            continue

        mass = Descriptors.ExactMolWt(mol)

        if not np.isfinite(mass) or mass <= 0:
            invalid += 1
            continue

        keys.append(key)
        smiles_list.append(smiles)
        exact_masses.append(mass)

    masses = np.asarray(exact_masses, dtype=np.float64)
    order = np.argsort(masses)

    keys = np.asarray(keys, dtype=object)[order]
    smiles_list = np.asarray(smiles_list, dtype=object)[order]
    masses = masses[order]

    print("Valid candidates:", len(keys))
    print("Invalid candidates:", invalid)

    return keys, smiles_list, masses


def main():
    print("Lupin V3 200k / 800-round inference")
    print("XGBoost version:", xgb.__version__)

    for path in [TEST_FILE, CATALOG_FILE, MODEL_FILE, FEATURE_FILE]:
        if not path.exists():
            raise FileNotFoundError(path)

    if RESULT_FILE.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {RESULT_FILE}"
        )

    model = xgb.XGBRanker()
    model.load_model(MODEL_FILE)

    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    model_features = model.get_booster().num_features()

    if len(feature_names) != 70 or model_features != 70:
        raise ValueError(
            f"Expected 70 features; JSON={len(feature_names)}, "
            f"model={model_features}"
        )

    print("Model:", MODEL_FILE.name)
    print("Boosted rounds:", model.get_booster().num_boosted_rounds())
    print("Features:", len(feature_names))

    if model.get_booster().num_boosted_rounds() != 800:
        raise ValueError("Model is not the expected 800-round model")

    print("\nLoading test spectra...")

    table = pq.read_table(TEST_FILE)
    data = table.to_pydict()

    total_spectra = table.num_rows

    print("Test spectra:", f"{total_spectra:,}")
    print("Test molecules:", len(set(data["molecule_id"])))

    catalog_keys, catalog_smiles, catalog_masses = load_catalog()

    candidate_cache = {}
    prediction_rows = []

    total_pairs = 0
    spectra_without_candidates = 0

    for i in range(total_spectra):
        spectrum_id = data["spectrum_id"][i]
        molecule_id = data["molecule_id"][i]

        spectrum = extract_features(
            precursor_mz=data["precursor_mz"][i],
            adduct=data["adduct"][i],
            mz_values=data["ms2_mzs"][i],
            intensities=data["ms2_normalized_intensities"][i],
            collision_energy=data["collision_energy_ev"][i],
            ionization_mode=data["ionization_mode"][i],
            instrument_type=data["instrument_type"][i],
        )

        neutral_mass = float(spectrum["neutral_mass"])

        if not np.isfinite(neutral_mass) or neutral_mass <= 0:
            raise ValueError(
                f"Invalid neutral mass for spectrum {spectrum_id}"
            )

        tolerance = neutral_mass * MASS_TOLERANCE_PPM / 1_000_000

        left = np.searchsorted(
            catalog_masses,
            neutral_mass - tolerance,
            side="left",
        )

        right = np.searchsorted(
            catalog_masses,
            neutral_mass + tolerance,
            side="right",
        )

        candidate_indices = np.arange(left, right)

        if len(candidate_indices) == 0:
            spectra_without_candidates += 1
            continue

        candidate_masses = catalog_masses[candidate_indices]

        mass_errors = (
            (neutral_mass - candidate_masses)
            / candidate_masses
            * 1_000_000
        )

        valid = np.abs(mass_errors) <= MASS_TOLERANCE_PPM

        candidate_indices = candidate_indices[valid]
        mass_errors = mass_errors[valid]

        if len(candidate_indices) == 0:
            spectra_without_candidates += 1
            continue

        feature_rows = []
        candidate_metadata = []

        for catalog_idx, mass_error in zip(
            candidate_indices,
            mass_errors,
        ):
            key = catalog_keys[catalog_idx]
            smiles = catalog_smiles[catalog_idx]

            if key not in candidate_cache:
                candidate_cache[key] = extract_molecular_features_v2(
                    str(smiles)
                )

            combined = dict(spectrum)
            combined.update(candidate_cache[key])

            combined["candidate_mass_error_ppm"] = float(mass_error)
            combined["candidate_abs_mass_error_ppm"] = float(
                abs(mass_error)
            )

            missing = [
                name for name in feature_names
                if name not in combined
            ]

            if missing:
                raise ValueError(
                    f"Missing features for {key}: {missing}"
                )

            feature_rows.append([
                combined[name] for name in feature_names
            ])

            candidate_metadata.append((key, smiles))

        X = np.asarray(feature_rows, dtype=np.float32)

        if X.shape[1] != 70:
            raise ValueError(f"Unexpected feature shape: {X.shape}")

        if not np.isfinite(X).all():
            raise ValueError(
                f"Non-finite features for spectrum {spectrum_id}"
            )

        scores = model.predict(X)

        for (key, smiles), score in zip(candidate_metadata, scores):
            prediction_rows.append({
                "molecule_id": molecule_id,
                "candidate_inchikey14": key,
                "candidate_smiles": smiles,
                "score": float(score),
            })

        total_pairs += len(scores)

        if (i + 1) % 100 == 0 or i + 1 == total_spectra:
            print(
                f"  Spectra processed: {i + 1:,}/{total_spectra:,} "
                f"| pairs: {total_pairs:,}"
            )

    print("\nInference completed.")
    print("Total candidate pairs:", f"{total_pairs:,}")
    print("Spectra without candidates:", spectra_without_candidates)

    if not prediction_rows:
        raise ValueError("No candidate predictions were generated")

    predictions = pd.DataFrame(prediction_rows)

    aggregated = (
        predictions
        .groupby(
            ["molecule_id", "candidate_inchikey14"],
            as_index=False,
        )
        .agg({
            "candidate_smiles": "first",
            "score": "mean",
        })
    )

    ranked = (
        aggregated
        .sort_values(
            ["molecule_id", "score"],
            ascending=[True, False],
        )
        .groupby("molecule_id", sort=False)
        .head(TOP_K)
        .reset_index(drop=True)
    )

    result = ranked[
        ["molecule_id", "candidate_smiles", "score"]
    ]

    if result["score"].isna().any():
        raise ValueError("NaN scores in final output")

    if result.duplicated(
        ["molecule_id", "candidate_smiles"]
    ).any():
        raise ValueError("Duplicate molecule-candidate pairs")

    result.to_csv(RESULT_FILE, index=False)

    print("\nV3 ranking saved:", RESULT_FILE)
    print("Output rows:", len(result))
    print("Output molecules:", result["molecule_id"].nunique())

    print("\nCandidates per molecule:")
    print(result.groupby("molecule_id").size().describe())

    print("\nTop 5 output rows:")
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()
