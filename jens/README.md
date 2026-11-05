# Jens Candidate Pipeline

A candidate-generation pipeline for mass-spectrometry-based molecule
identification. Jens's component finds plausible known natural-product
molecules for query spectra and supplies candidates/evidence to the
team's downstream ranking stage.

> **Status: Prototype / in progress.** Validate input schemas, structure
> standardization, adduct handling, spectral-library integration, and
> final submission format against the competition data and its
> `metric.py`.

## Responsibilities

-   Build a known natural-product collection from permitted sources such
    as COCONUT, LOTUS, and NPAtlas.
-   Standardize and deduplicate structures using the competition's
    required procedure.
-   Generate candidates by precursor-mass matching, accounting for
    adduct/charge metadata where available.
-   Match query spectra against permitted reference libraries such as
    GNPS or MassBank when compatible local files are available.
-   Export candidate records and evidence for downstream ranking.

## Project structure

``` text
jens_candidate_pipeline/
├── data/
│   ├── raw/
│   │   ├── coconut/          # Source data (not committed)
│   │   ├── lotus/            # Source data (not committed)
│   │   ├── npatlas/          # Source data (not committed)
│   │   └── competition/      # Train/test data (not committed)
│   ├── processed/             # Processed molecules (not committed)
│   └── spectra_libraries/
│       ├── gnps/              # Reference spectra (not committed)
│       └── massbank/          # Reference spectra (not committed)
├── outputs/                   # Generated results (not committed)
├── src/
│   ├── inspect_data.py
│   ├── prepare_database.py
│   ├── build_mass_index.py
│   ├── generate_candidates.py
│   └── match_spectra.py
├── requirements.txt
└── README.md
```

Adjust filenames to match the actual scripts and datasets in your local
repository.

## Setup (macOS / VS Code)

Open the repository root in VS Code and run:

``` bash
python3 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install pandas numpy rdkit pyarrow matchms
```

Optionally record installed package versions:

``` bash
python -m pip freeze > requirements-lock.txt
```

## Data sources

Download only data allowed by the competition rules. Keep original
formats when possible. Check licenses, attribution, and redistribution
restrictions before use or publication.

-   [COCONUT](https://coconut.naturalproducts.net/)
-   [LOTUS](https://lotus.naturalproducts.net/)
-   [NPAtlas](https://www.npatlas.org/)
-   [GNPS](https://gnps.ucsd.edu/)
-   [MassBank](https://massbank.eu/MassBank/)

Place source files under `data/raw/` and reference spectra under
`data/spectra_libraries/`. Filenames and formats can vary by dataset
release. Do not commit large datasets, competition files, or third-party
data unless permitted.

## Workflow

Run from the repository root with `.venv` activated. Confirm each
script's actual arguments and configured filenames before executing.

``` bash
python src/inspect_data.py
python src/prepare_database.py
python src/build_mass_index.py
python src/generate_candidates.py
```

### Spectral library matching (MGF baseline)

The baseline matcher expects query and reference spectra in MGF format.
Example:

``` bash
python src/match_spectra.py \
  --query data/raw/competition/test.mgf \
  --reference data/spectra_libraries/gnps/ALL_GNPS.mgf \
  --output outputs/spectral_matches.csv \
  --top-n 25
```

Replace paths with your actual local filenames. The baseline filters by
precursor mass and compares fragment peaks using cosine similarity.
Large libraries may require optimization.

## Expected outputs

Outputs depend on script configuration and successful execution. The
workflow may produce:

-   `data/processed/molecule_database.parquet` --- standardized molecule
    database.
-   `data/processed/mass_index.parquet` --- mass-sorted molecule index.
-   `outputs/candidates.csv` --- candidates generated for query spectra.
-   `outputs/candidate_evidence.csv` --- candidate supporting evidence.
-   `outputs/spectral_matches.csv` --- reference spectral matches.

Treat these as expected names; verify the filenames in the scripts.

## Shared candidate interface

The team proposed a shared record format:

``` text
molecule_id, candidate_smiles, score
```

Confirm with the ranking teammate what `molecule_id` refers to and
whether a query/spectrum linkage column is required. Scores from this
stage are preliminary; downstream ranking should combine candidate
sources and produce the final order.

## Validation checklist

-   [ ] Confirm source datasets comply with competition rules.
-   [ ] Inspect real data columns and formats.
-   [ ] Match canonicalization exactly to competition `metric.py` (RDKit
    version, stereochemistry, tautomer handling, InChIKey14).
-   [ ] Handle adduct and charge using available metadata.
-   [ ] Deduplicate by the required canonical key.
-   [ ] Evaluate candidate coverage / Recall@K using a leakage-safe
    validation split.
-   [ ] Report separately results for database-covered and
    database-missing molecules.
-   [ ] Run official evaluation on final output.
-   [ ] Test the pipeline offline.
-   [ ] Measure runtime and keep the workflow within the 9-hour limit.

## Limitations

-   Precursor-mass matching can return isobaric candidates; it does not
    establish identity.
-   Precursor m/z alone is insufficient for reliable neutral-mass
    conversion without adduct and charge assumptions.
-   The MGF spectral matcher is a baseline and does not automatically
    merge matches into the candidate CSV.
-   Evaluation compatibility is unverified until the actual `metric.py`
    and input schema are tested.
-   Library matching can only help when relevant reference compounds are
    present in the library.

## Team integration

Jens supplies candidates and evidence to Lupin's ranking stage. Database
matches, spectral-library matches, and de novo-generated candidates
should be consolidated into one list and ranked downstream. Alex handles
validation and final submission assembly as described in the team plan.

## License and data notice

Choose and add a software license for this repository before public
release. Record dataset licenses and citations (for example in
`DATA_SOURCES.md`). Do not redistribute external datasets or competition
data unless their terms explicitly allow it.

## References

-   [COCONUT](https://coconut.naturalproducts.net/)
-   [LOTUS](https://lotus.naturalproducts.net/)
-   [NPAtlas](https://www.npatlas.org/)
-   [GNPS](https://gnps.ucsd.edu/)
-   [MassBank](https://massbank.eu/MassBank/)
