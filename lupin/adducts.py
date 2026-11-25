
"""
Adduct-aware precursor mass calculations for Lupin.

Converts observed precursor m/z into estimated neutral molecular mass.
Supports all seven adducts found in the CASMI 2026 test set.
"""

from dataclasses import dataclass
import math


# Monoisotopic mass constants (Da)
PROTON = 1.007276466621
ELECTRON = 0.000548579909
SODIUM = 22.9897692820
POTASSIUM = 38.9631580
CHLORINE = 34.968852682
NITROGEN = 14.003074004
HYDROGEN = 1.007825032

FORMIC_ACID = (
    12.0
    + 2 * HYDROGEN
    + 2 * 15.994914620
)

# Each mass_shift is the mass added to neutral M
# to produce the observed ion.
@dataclass(frozen=True)
class Adduct:
    mass_shift: float
    charge: int
    polarity: str


ADDUCTS = {
    "[M+H]+": Adduct(
        mass_shift=PROTON,
        charge=1,
        polarity="positive",
    ),

    "[M-H]-": Adduct(
        mass_shift=-PROTON,
        charge=-1,
        polarity="negative",
    ),

    "[M+CH2O2-H]-": Adduct(
        mass_shift=FORMIC_ACID - PROTON,
        charge=-1,
        polarity="negative",
    ),

    "[M+Na]+": Adduct(
        mass_shift=SODIUM - ELECTRON,
        charge=1,
        polarity="positive",
    ),

    "[M+NH4]+": Adduct(
        mass_shift=NITROGEN + 4 * HYDROGEN - ELECTRON,
        charge=1,
        polarity="positive",
    ),

    "[M+K]+": Adduct(
        mass_shift=POTASSIUM - ELECTRON,
        charge=1,
        polarity="positive",
    ),

    "[M+Cl]-": Adduct(
        mass_shift=CHLORINE + ELECTRON,
        charge=-1,
        polarity="negative",
    ),
}


def neutral_mass_from_mz(mz: float, adduct: str) -> float:
    """
    Calculate neutral molecular mass from precursor m/z.

    Formula:
        neutral_mass = mz * abs(charge) - mass_shift

    Args:
        mz: Observed precursor mass-to-charge ratio.
        adduct: Adduct notation, e.g. '[M+H]+'.

    Returns:
        Estimated neutral molecular mass in Da.
    """

    if adduct not in ADDUCTS:
        raise ValueError(f"Unsupported adduct: {adduct}")

    if not math.isfinite(mz) or mz <= 0:
        raise ValueError(f"Invalid precursor m/z: {mz}")

    info = ADDUCTS[adduct]

    neutral_mass = (
        mz * abs(info.charge) - info.mass_shift
    )

    if neutral_mass <= 0:
        raise ValueError(
            f"Calculated neutral mass is invalid: {neutral_mass}"
        )

    return neutral_mass


def ppm_error(observed: float, theoretical: float) -> float:
    """
    Calculate signed mass error in parts per million.
    """

    if not math.isfinite(observed) or not math.isfinite(theoretical):
        raise ValueError("Mass values must be finite.")

    if observed <= 0 or theoretical <= 0:
        raise ValueError("Mass values must be positive.")

    return (observed - theoretical) / theoretical * 1_000_000


def within_ppm_tolerance(
    observed: float,
    theoretical: float,
    tolerance: float = 10.0,
) -> bool:
    """Check whether two masses match within a ppm tolerance."""

    if tolerance < 0:
        raise ValueError("Tolerance cannot be negative.")

    return abs(ppm_error(observed, theoretical)) <= tolerance


def test_adducts():
    """Basic round-trip tests for all supported adducts."""

    known_mass = 300.123456

    print("Testing adduct mass calculations...\n")

    for name, info in ADDUCTS.items():

        mz = (
            known_mass + info.mass_shift
        ) / abs(info.charge)

        recovered_mass = neutral_mass_from_mz(mz, name)

        assert math.isclose(
            recovered_mass,
            known_mass,
            rel_tol=0,
            abs_tol=1e-9,
        ), f"Failed: {name}"

        print(
            f"{name:18s} "
            f"m/z={mz:12.6f} "
            f"neutral={recovered_mass:.6f} "
            f"PASS"
        )

    print("\nAll seven adduct tests passed!")


if __name__ == "__main__":
    test_adducts()
