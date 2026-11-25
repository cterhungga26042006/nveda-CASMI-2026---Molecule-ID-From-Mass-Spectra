"""
Lupin - MS/MS spectrum feature extraction.
"""

import numpy as np

from adducts import ADDUCTS, neutral_mass_from_mz


ADDUCT_NAMES = sorted(ADDUCTS.keys())


def normalize_instrument(value):
    """Group similar instrument names."""

    if value is None:
        return "unknown"

    value = str(value).strip().lower()

    if value in ("", "nan", "n/a", "none"):
        return "unknown"

    if "tims" in value:
        return "timsTOF"
    if "orbitrap" in value or "qexactive" in value:
        return "Orbitrap"
    if "qtof" in value or "q-tof" in value:
        return "QTOF"
    if "tof" in value:
        return "TOF"
    if "qqq" in value or "qq" in value:
        return "TripleQuadrupole"
    if "ion trap" in value or "iontrap" in value:
        return "IonTrap"
    if "ft" in value:
        return "FT"
    if "maldi" in value:
        return "MALDI"
    if "esi" in value:
        return "ESI"
    if "apci" in value:
        return "APCI"

    return "other"


def safe_array(values):
    """Convert values to a finite NumPy array."""

    if values is None:
        return np.array([], dtype=np.float64)

    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return np.array([], dtype=np.float64)

    return array[np.isfinite(array)]


def extract_features(
    precursor_mz,
    adduct,
    mz_values,
    intensities,
    collision_energy=None,
    ionization_mode=None,
    instrument_type=None,
):
    """Extract numerical features from one spectrum."""

    mz = safe_array(mz_values)
    intensity = safe_array(intensities)

    if len(mz) != len(intensity):
        raise ValueError("m/z and intensity lengths do not match")

    precursor_mz = float(precursor_mz)

    features = {}

    # Precursor
    features["precursor_mz"] = precursor_mz
    features["neutral_mass"] = neutral_mass_from_mz(
        precursor_mz, adduct
    )

    # Adduct
    for name in ADDUCT_NAMES:
        features[f"adduct_{name}"] = int(adduct == name)

    # Ionization mode
    features["ionization_positive"] = int(
        ionization_mode == "positive"
    )
    features["ionization_negative"] = int(
        ionization_mode == "negative"
    )

    # Instrument
    instrument = normalize_instrument(instrument_type)

    categories = [
        "timsTOF", "Orbitrap", "QTOF", "TOF",
        "TripleQuadrupole", "IonTrap", "FT",
        "MALDI", "ESI", "APCI", "unknown", "other"
    ]

    for category in categories:
        features[f"instrument_{category}"] = int(
            instrument == category
        )

    # Collision energy
    energy = safe_array(collision_energy)

    features["collision_energy_missing"] = int(len(energy) == 0)
    features["collision_energy_mean"] = (
        float(np.mean(energy)) if len(energy) else 0.0
    )
    features["collision_energy_max"] = (
        float(np.max(energy)) if len(energy) else 0.0
    )
    features["collision_energy_min"] = (
        float(np.min(energy)) if len(energy) else 0.0
    )

    # Spectrum size
    features["num_peaks"] = len(mz)

    if len(mz) == 0:
        features.update({
            "intensity_sum": 0.0,
            "intensity_max": 0.0,
            "intensity_mean": 0.0,
            "intensity_std": 0.0,
            "intensity_weighted_mz": 0.0,
            "fragment_mz_min": 0.0,
            "fragment_mz_max": 0.0,
            "fragment_mz_mean": 0.0,
            "fragment_mz_std": 0.0,
            "fragment_to_precursor_ratio": 0.0,
        })
        return features

    # Intensity statistics
    intensity = np.maximum(intensity, 0.0)

    features["intensity_sum"] = float(np.sum(intensity))
    features["intensity_max"] = float(np.max(intensity))
    features["intensity_mean"] = float(np.mean(intensity))
    features["intensity_std"] = float(np.std(intensity))

    # Fragment m/z statistics
    features["fragment_mz_min"] = float(np.min(mz))
    features["fragment_mz_max"] = float(np.max(mz))
    features["fragment_mz_mean"] = float(np.mean(mz))
    features["fragment_mz_std"] = float(np.std(mz))

    # Intensity-weighted m/z
    intensity_sum = np.sum(intensity)

    if intensity_sum > 0:
        weighted_mz = np.sum(mz * intensity) / intensity_sum
    else:
        weighted_mz = 0.0

    features["intensity_weighted_mz"] = float(weighted_mz)

    # Fragment-to-precursor ratio
    features["fragment_to_precursor_ratio"] = float(
        np.max(mz) / precursor_mz
    )

    return features
