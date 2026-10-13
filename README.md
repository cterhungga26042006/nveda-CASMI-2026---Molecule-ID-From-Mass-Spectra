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

## 1. Setup (Windows + WSL)

### 1.1 Clone the repo

```bash
git clone https://github.com/cterhungga26042006/nveda-CASMI-2026---Molecule-ID-From-Mass-Spectra.git
cd nveda-CASMI-2026---Molecule-ID-From-Mass-Spectra
```

Tip: in WSL, work inside the Linux home (for example `~/Enveda`) rather than `/mnt/c/...`. It is much faster.

### 1.2 Create the virtual environment

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip unzip   # first time only
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install kaggle
```

Install project packages (once `requirements.txt` exists):

```bash
pip install -r requirements.txt
```

Useful packages for this project: `numpy`, `pandas`, `rdkit`, `torch`, `scikit-learn`, `matchms`, `jupyter`.

Leave the environment with `deactivate`. Re-enter it with `source .venv/bin/activate` every time you open a new terminal.

### 1.3 Set up your Kaggle API token

1. Go to kaggle.com, profile picture, **Settings**, **API**, **Create New Token**.
2. Place the downloaded `kaggle.json` in `~/.kaggle/`:

```bash
mkdir -p ~/.kaggle
cp "/mnt/d/Users/<you>/Downloads/kaggle.json" ~/.kaggle/kaggle.json   # adjust the path
chmod 600 ~/.kaggle/kaggle.json
```

3. Test it:

```bash
kaggle competitions list | head
```

Keep the token private. Do not share it or commit it.

### 1.4 Download the competition data

Accept the rules on the competition page first, otherwise you get a 403 error.

```bash
kaggle competitions files enveda-CASMI26-molecule-id-mass-spectra   # check access
kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra
unzip enveda-CASMI26-molecule-id-mass-spectra.zip -d data/
ls -lh data/
```

You can also download the files from the competition's **Data** tab and