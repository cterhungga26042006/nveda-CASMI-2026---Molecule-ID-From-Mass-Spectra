# Enveda CASMI 2026 - Molecule ID From Mass Spectra

Team repository for the Kaggle competition
[enveda-CASMI26-molecule-id-mass-spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra).

**Goal:** predict the 2D structure (a SMILES string) of a molecule from its LC-MS/MS spectrum, including molecules never seen before.

> **Code competition:** the final `submission.csv` must be produced by a Kaggle Notebook (CPU or GPU, max 9 hours, internet disabled). Training can happen anywhere, but the weights must be uploaded to Kaggle as a Dataset/Model.

---

## Key dates (11:59 PM UTC)

| Date | Event |
|---|---|
| Sep 14, 2026 | Start |
| **Dec 7, 2026** | Entry deadline and team merger deadline |
| **Dec 14, 2026** | Final submission deadline |

Prizes: 1st $16,000, 2nd $12,000, 3rd $9,000, 4th $7,000, 5th $6,000.

---

## Rules to remember

- Every teammate must click **Join Competition** and accept the rules with their own Kaggle account.
- Merge Kaggle accounts into one team before **Dec 7**.
- **Never commit** competition data, `kaggle.json`, API keys, or large model files to this repo.
- Never put competition data in a public repo or public Kaggle Dataset.
- Submission file must be named `submission.csv`.
- Free, publicly available external data and pretrained models are allowed.

---

## Evaluation

Submissions are scored with Mean Reciprocal Rank @ 25 (MRR @25):

**Formula:** MRR @25 = (1/U) * ∑(1/rank_u) for all U molecules
- U is the number of molecules.
- rank_u is the position of the first correct structure in your ranked list for molecule u.
- A molecule scores 0 if none of its guesses is correct.

**Scoring:** Each molecule has exactly one correct structure, so only the first correct guess counts.
- A correct guess at position 1 scores 1.0, at position 2 scores 0.5, and at position 25 scores 0.04.

### Matching Rules

A prediction is correct when it has the same atom connectivity as the answer:
- Both the prediction and the answer are passed through RDKit's tautomer canonicalization (pinned at RDKit 2026.03.3) and reduced to the first block of their InChIKey (InChIKey14).
- The two keys must match.

**Note:** Stereochemistry is removed before tautomer canonicalization. Stereocenters and tautomer forms are not penalized.

### Submission Tips

- Duplicate guesses take up rank positions, so remove duplicates before submitting.
- An invalid SMILES counts as a wrong guess and still takes up a rank position.
- A molecule that is missing from the submission scores 0.

Official metric notebook: https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank

---

## Code Requirements

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

- CPU Notebook <= 9 hours run-time
- GPU Notebook <= 9 hours run-time
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models
- Submission file must be named `submission.csv`

Please see the [Code Competition FAQ](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/code-competition-faq) for more information on how to submit. And review the [code debugging doc](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/code-debugging-doc) if you are encountering submission errors.

---
