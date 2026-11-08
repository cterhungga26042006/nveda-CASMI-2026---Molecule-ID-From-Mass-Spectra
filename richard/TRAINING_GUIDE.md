# Training guide (Kaggle)

## Setup
1. Join the competition with your own Kaggle account.
2. Create a new notebook and add the competition as input.
3. Accelerator: GPU T4. Internet: on (cell 1 runs pip install).
4. Stop any other GPU session first; Kaggle allows only one interactive GPU session.

## Notebook layout (10 cells)
1. pip install rdkit lightning tokenizers
2. Imports and paths. Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
3. Read train.parquet, split train/val by inchikey14, process spectra.
4. Train the BPE tokenizer and save tokenizer.json.
5. Datamodule. VAL_BATCH_SIZE = 16.
6. PeakEmbedder, SpectrumEncoder and the fixed SmilesDecoder (max_new_tokens=100).
7. Lightning model.
8. Model parameters and a smoke test (1 train batch + 1 validation batch).
9. Training: max_epochs=3, accumulate_grad_batches=3, gradient_clip_val=1.0, LR 1e-4.
10. Save metrics.json, denovo_weights.pt and final.ckpt.

## How to run: use a background commit, not an interactive session
An interactive session dies when the browser disconnects or the computer sleeps, and
/kaggle/working is then wiped. One 4-hour run was lost this way.
1. Run cells 1-8 interactively until SMOKE TEST OK appears (a few minutes).
2. Save Version -> Save & Run All (Commit), GPU T4, Internet on.
3. You can then close the browser or shut down. The log shows a 12-hour limit per run
   (timeout=43200).

## Observed timing
The successful run was marked Failed after 16,613 s (about 4 h 37 min) but had saved the
epoch-2 checkpoint. Each epoch has 1563 batches, which is 521 optimizer steps with
gradient accumulation 3.

## Reading the logs
The Lightning progress bar and train_loss do not appear in the Logs of a background commit;
only errors and print() output do. No Traceback means the run has not crashed.
The message "Could not find cuda drivers" comes from TensorFlow (imported by TensorBoard),
not from PyTorch, and can be ignored.

## Problems encountered
- CUDA out of memory during validation: batch 128 x 10 samples = 1280 sequences and
  generate() has no KV cache. Fixed with VAL_BATCH_SIZE=16 and max_new_tokens=100.
- Loss stuck at about 4.6: LR 3e-4 without warmup collapsed training. Use LR 1e-4,
  5% warmup, gradient clipping 1.0.
- NameError after a kernel restart: all variables are lost; rerun the cells from the top.
- Final cell did not produce its files: cause not determined. Take the checkpoint from
  model_checkpoints/ in the run's Output tab instead.

## Outputs to download after a run
- model_checkpoints/denovo_fixed/epoch=2-step=1563-val_tanimoto=....ckpt
- tokenizer.json
- last.ckpt (optional)
Never commit these to GitHub.

## Evaluating a checkpoint (separate notebook)
1. New notebook, GPU T4. Add the competition and the training notebook's output as inputs.
2. Run cells 1-8 without the smoke test. In cell 4, load tokenizer.json with
   Tokenizer.from_file instead of retraining it, otherwise token IDs will not match.
3. torch.load the checkpoint; load_state_dict must report "All keys matched successfully".
4. Run validation and read valid_smiles, tanimoto and exact match.

## Next steps
score(spectrum, smiles) function, candidate table for Lupin, MRR@25 under metric.py.
