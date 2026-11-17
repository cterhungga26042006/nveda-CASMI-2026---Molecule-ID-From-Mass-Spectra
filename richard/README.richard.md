# De novo molecule generation (Richard)

Status: baseline v1. Decoder bugs are fixed and the model learns SMILES syntax, but it is
undertrained and does not yet recover structures from spectra. Not ready as a ranking source.

## Goal
Generate SMILES structures directly from MS/MS spectra, including molecules not in any
database. Candidates are meant to go to Lupin for ranking together with Jens's database
candidates. De novo output is expected to be a fallback, rarely rank 1.

## Model
- Peak embedder (sinusoidal m/z + intensity) -> 6-layer spectrum encoder -> 6-layer SMILES
  decoder, BPE vocabulary of 512 tokens, about 103M parameters.
- Train/validation split by inchikey14: the 500 validation structures never appear in training.
- Trained on 200,000 spectra (of about 1.39M), 3 epochs. AdamW, LR 1e-4, weight decay 0.01,
  5% warmup then cosine decay, gradient clipping 1.0, bf16 mixed precision, batch 128 with
  3x gradient accumulation (effective 384).

## Bugs fixed relative to the tutorial notebook
1. Labels were not shifted by one token, so the model learned to copy its input.
2. The causal mask was applied only in training, not in generation.
3. The decoder had no positional embedding.
4. generate() fed only the last token instead of the full generated prefix.
Also: validation batch size 16 and max_new_tokens=100 (longest training SMILES is 99 tokens),
because 128 x 10 samples without a KV cache ran out of GPU memory.

## Results (500 held-out structures, 1 spectrum each, 10 samples per spectrum)
| Metric | Tutorial as given (1 epoch) | This repo (3 epochs) |
|---|---|---|
| val_valid_smiles | 0.066 | 0.848 |
| val_tanimoto | 0.002 | 0.077 |
| val_exact match | 0.0 | 0.0 |

Definitions: valid_smiles is the fraction of spectra with at least one valid SMILES among
10 samples. tanimoto is computed for the highest-likelihood valid sample (0 if none).
These are not the competition metric (MRR@25), so they are not directly comparable to it.
The two columns also differ in training length, so the comparison is not controlled.

Qualitative check on 8 validation examples: predictions are mostly common lipid and sugar
structures from the training set and do not reflect the input spectrum.

## Training run
The Kaggle commit run was marked Failed after 16,613 s, but the epoch-2 checkpoint
(step 1563, val_tanimoto 0.082) was saved. The cause of the failure was not investigated
(Logs not inspected); it is suspected to be in the final cell that saves metrics.

## Files
- TRAINING_GUIDE.md: how to reproduce the training on Kaggle.
- denovo_train.ipynb: training notebook (outputs cleared).
- denovo_eval.ipynb: loads the checkpoint and evaluates it.

## Checkpoint
Not in this repo (1.24 GB). It is stored in a private Kaggle dataset:
https://www.kaggle.com/datasets/richardnguyen1412/denovo-v1-weights
together with tokenizer.json. The same tokenizer.json must be used with the checkpoint; aretrained tokenizer gives different token IDs.

## Not done yet
- score(spectrum, smiles) function (teacher-forced log-likelihood) for Lupin.
- Candidate table in the shared format (molecule_id, candidate_smiles, score).
- MRR@25 under metric.py; metric.py has not been read yet.
- Check whether the test set has an adduct column, and filter candidates by precursor mass.
- Prediction runtime on the test set with internet disabled.

## Limitations
Only 3 epochs on 200k spectra; exact match is 0. Evaluated with RDKit 2026.9.1 on Kaggle,
not checked against the RDKit version used by metric.py.
