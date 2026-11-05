
#!/usr/bin/env python3
"""
Match query MS/MS spectra against a reference spectral library in MGF format.

Outputs:
    outputs/spectral_matches.csv

Assumptions:
    - Both query and reference spectra are MGF files.
    - PEPMASS contains the precursor m/z.
    - Fragment peaks are represented as: m/z intensity.
    - Reference metadata may contain NAME, SMILES, INCHIKEY, or SCANS.

This is a baseline spectral-matching tool, not the final competition
candidate-generation or MRR@25 evaluation pipeline.
"""

import argparse
import csv
import math
import re
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_mgf(path):
    """Read MGF spectra into a list of dictionaries."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"MGF file not found: {path}")

    spectra = []
    current = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if line.upper() == "BEGIN IONS":
                current = {"metadata": {}, "peaks": []}
                continue

            if line.upper() == "END IONS":
                if current is not None:
                    spectrum = finish_spectrum(current, len(spectra) + 1)
                    if spectrum is not None:
                        spectra.append(spectrum)
                current = None
                continue

            if current is None:
                continue

            # Metadata lines are generally KEY=VALUE.
            if "=" in line:
                key, value = line.split("=", 1)
                current["metadata"][key.strip().upper()] = value.strip()
                continue

            # Fragment peak line: m/z intensity [optional extra values].
            fields = line.split()
            if len(fields) >= 2:
                try:
                    mz = float(fields[0])
                    intensity = float(fields[1])

                    if (
                        math.isfinite(mz)
                        and math.isfinite(intensity)
                        and mz > 0
                        and intensity > 0
                    ):
                        current["peaks"].append((mz, intensity))
                except ValueError:
                    # Ignore non-peak lines that do not parse as numbers.
                    continue

    return spectra


def first_value(metadata, keys, default=""):
    """Return the first non-empty metadata value among alternative keys."""
    for key in keys:
        value = metadata.get(key.upper(), "")
        if value:
            return value
    return default


def finish_spectrum(raw, fallback_id):
    """Validate a spectrum and extract its commonly used metadata."""
    metadata = raw["metadata"]
    peaks = raw["peaks"]

    pepmass_text = metadata.get("PEPMASS", "")
    precursor_mz = None

    if pepmass_text:
        try:
            # PEPMASS may contain both m/z and precursor intensity.
            precursor_mz = float(pepmass_text.split()[0])
        except (ValueError, IndexError):
            precursor_mz = None

    if precursor_mz is None or precursor_mz <= 0 or not peaks:
        return None

    spectrum_id = first_value(
        metadata,
        ["SCANS", "SPECTRUMID", "FEATURE_ID", "TITLE"],
        default=f"spectrum_{fallback_id}",
    )

    name = first_value(
        metadata,
        ["NAME", "COMPOUND_NAME", "COMPOUND", "TITLE"],
    )

    smiles = first_value(
        metadata,
        ["SMILES", "CANONICAL_SMILES", "ISOMERICSMILES"],
    )

    inchikey = first_value(
        metadata,
        ["INCHIKEY", "INCHI_KEY", "INCHIAUX"],
    )

    # Merge duplicate fragment m/z values by summing their intensities.
    merged = {}
    for mz, intensity in peaks:
        merged[mz] = merged.get(mz, 0.0) + intensity

    clean_peaks = sorted(merged.items())

    return {
        "spectrum_id": str(spectrum_id),
        "name": name,
        "smiles": smiles,
        "inchikey": inchikey,
        "precursor_mz": precursor_mz,
        "peaks": clean_peaks,
        "metadata": metadata,
    }


def normalize_peaks(peaks):
    """
    Square-root transform intensities and L2-normalize them.

    This reduces the dominance of very intense peaks.
    """
    transformed = [(mz, math.sqrt(intensity)) for mz, intensity in peaks]
    norm = math.sqrt(sum(intensity**2 for _, intensity in transformed))

    if norm == 0:
        return []

    return [(mz, intensity / norm) for mz, intensity in transformed]


def cosine_similarity(query_peaks, reference_peaks, fragment_tolerance):
    """
    Calculate a greedy one-to-one fragment-peak cosine similarity.

    Peaks must be within fragment_tolerance Da to be matched.
    Returns (score, number_of_matched_peak_pairs).
    """
    q_peaks = normalize_peaks(query_peaks)
    r_peaks = normalize_peaks(reference_peaks)

    if not q_peaks or not r_peaks:
        return 0.0, 0

    possible_matches = []

    for qi, (q_mz, q_intensity) in enumerate(q_peaks):
        for ri, (r_mz, r_intensity) in enumerate(r_peaks):
            error = abs(q_mz - r_mz)

            if error <= fragment_tolerance:
                contribution = q_intensity * r_intensity
                possible_matches.append(
                    (contribution, error, qi, ri)
                )

    # Match the strongest peak pairs first, using each peak at most once.
    possible_matches.sort(reverse=True)

    used_query = set()
    used_reference = set()
    score = 0.0
    matched_pairs = 0

    for contribution, error, qi, ri in possible_matches:
        if qi in used_query or ri in used_reference:
            continue

        used_query.add(qi)
        used_reference.add(ri)
        score += contribution
        matched_pairs += 1

    return min(score, 1.0), matched_pairs


def ppm_error(observed_mz, reference_mz):
    """Signed ppm difference, relative to the reference m/z."""
    return (observed_mz - reference_mz) / reference_mz * 1_000_000


def find_matches(
    query_spectra,
    reference_spectra,
    precursor_tolerance_ppm=10.0,
    fragment_tolerance=0.02,
    top_n=25,
    min_cosine=0.0,
):
    """Find the best precursor-compatible spectral matches for each query."""
    references = sorted(
        reference_spectra,
        key=lambda spectrum: spectrum["precursor_mz"],
    )

    reference_masses = [
        spectrum["precursor_mz"] for spectrum in references
    ]

    results = []

    for query_number, query in enumerate(query_spectra, start=1):
        q_mz = query["precursor_mz"]
        tolerance_da = q_mz * precursor_tolerance_ppm / 1_000_000

        low = bisect_left(reference_masses, q_mz - tolerance_da)
        high = bisect_right(reference_masses, q_mz + tolerance_da)

        candidates = references[low:high]
        scored = []

        for reference in candidates:
            ref_mz = reference["precursor_mz"]
            mass_error = ppm_error(q_mz, ref_mz)

            cosine, matched_peaks = cosine_similarity(
                query["peaks"],
                reference["peaks"],
                fragment_tolerance,
            )

            if cosine < min_cosine:
                continue

            scored.append(
                {
                    "query_spectrum_id": query["spectrum_id"],
                    "reference_spectrum_id": reference["spectrum_id"],
                    "reference_name": reference["name"],
                    "reference_smiles": reference["smiles"],
                    "reference_inchikey": reference["inchikey"],
                    "query_precursor_mz": q_mz,
                    "reference_precursor_mz": ref_mz,
                    "precursor_error_ppm": mass_error,
                    "cosine_score": cosine,
                    "matched_peaks": matched_peaks,
                    "query_peak_count": len(query["peaks"]),
                    "reference_peak_count": len(reference["peaks"]),
                }
            )

        scored.sort(
            key=lambda row: (
                row["cosine_score"],
                row["matched_peaks"],
                -abs(row["precursor_error_ppm"]),
            ),
            reverse=True,
        )

        for rank, row in enumerate(scored[:top_n], start=1):
            row["rank"] = rank
            results.append(row)

        if query_number % 100 == 0:
            print(
                f"Processed {query_number}/{len(query_spectra)} "
                f"query spectra..."
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Match query MGF spectra against a reference MGF library."
    )

    parser.add_argument(
        "--query",
        default="data/raw/competition/test.mgf",
        help="Path to query/test spectra in MGF format.",
    )
    parser.add_argument(
        "--reference",
        default="data/spectra_libraries/gnps/ALL_GNPS.mgf",
        help="Path to reference spectra in MGF format.",
    )
    parser.add_argument(
        "--output",
        default="outputs/spectral_matches.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--precursor-ppm",
        type=float,
        default=10.0,
        help="Maximum precursor mass difference in ppm (default: 10).",
    )
    parser.add_argument(
        "--fragment-tolerance",
        type=float,
        default=0.02,
        help="Fragment peak tolerance in Da (default: 0.02).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Maximum matches to save per query (default: 25).",
    )
    parser.add_argument(
        "--min-cosine",
        type=float,
        default=0.0,
        help="Minimum cosine score to retain (default: 0).",
    )

    args = parser.parse_args()

    if args.precursor_ppm <= 0:
        parser.error("--precursor-ppm must be greater than zero.")
    if args.fragment_tolerance <= 0:
        parser.error("--fragment-tolerance must be greater than zero.")
    if args.top_n < 1:
        parser.error("--top-n must be at least 1.")
    if not 0 <= args.min_cosine <= 1:
        parser.error("--min-cosine must be between 0 and 1.")

    query_path = Path(args.query)
    reference_path = Path(args.reference)
    output_path = Path(args.output)

    if not query_path.is_absolute():
        query_path = PROJECT_ROOT / query_path
    if not reference_path.is_absolute():
        reference_path = PROJECT_ROOT / reference_path
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    try:
        print(f"Reading query spectra: {query_path}")
        query_spectra = parse_mgf(query_path)

        print(f"Reading reference spectra: {reference_path}")
        reference_spectra = parse_mgf(reference_path)

        if not query_spectra:
            raise ValueError(
                "No usable query spectra found. Check the file format "
                "and confirm it contains PEPMASS and fragment peaks."
            )

        if not reference_spectra:
            raise ValueError(
                "No usable reference spectra found. Check the MGF file."
            )

        print(f"Query spectra loaded: {len(query_spectra)}")
        print(f"Reference spectra loaded: {len(reference_spectra)}")

        results = find_matches(
            query_spectra,
            reference_spectra,
            precursor_tolerance_ppm=args.precursor_ppm,
            fragment_tolerance=args.fragment_tolerance,
            top_n=args.top_n,
            min_cosine=args.min_cosine,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        columns = [
            "query_spectrum_id",
            "rank",
            "reference_spectrum_id",
            "reference_name",
            "reference_smiles",
            "reference_inchikey",
            "query_precursor_mz",
            "reference_precursor_mz",
            "precursor_error_ppm",
            "cosine_score",
            "matched_peaks",
            "query_peak_count",
            "reference_peak_count",
        ]

        with output_path.open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)

        print(f"Matches saved to: {output_path}")
        print(f"Total match rows: {len(results)}")

        if not results:
            print(
                "No matches were found. Try checking the precursor tolerance, "
                "file metadata, or whether the spectra are compatible."
            )

    except (FileNotFoundError, PermissionError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
