"""Local helpers that mirror the official CASMI metric (metric.py).

Use these to canonicalize, de-duplicate and score candidates exactly as the competition does.
They reproduce the logic of the metric notebook:

  * strip stereochemistry, THEN tautomer-canonicalize, THEN InChIKey, keep the first 14 characters
  * a guess scores 1/rank (rank = position in the list, 1-based); only the first match counts
  * a molecule with no match scores 0; the final score is the mean over molecules
  * at most max_rank (25) guesses per molecule, joined with ';', best first

RDKit must be 2026.3.3: the metric states that TautomerEnumerator output changes between
releases, so a different version can canonicalize the same molecule differently.
"""

import warnings

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

import rdkit

EXPECTED_RDKIT_VERSION = '2026.03.3'
if rdkit.__version__ != EXPECTED_RDKIT_VERSION:
    warnings.warn(
        f'RDKit {rdkit.__version__} != {EXPECTED_RDKIT_VERSION} used by the metric; '
        'canonicalization may differ from the official scorer.', UserWarning)

RDLogger.DisableLog('rdApp.*')
_taut = rdMolStandardize.TautomerEnumerator()


def canonicalize_smiles_to_ik14(smiles):
    """InChIKey14 of the stereo-stripped, tautomer-canonical structure, or None if invalid."""
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToInchiKey(_taut.Canonicalize(mol))[:14]
    except (ValueError, RuntimeError, TypeError):
        return None


def heavy_atom_composition(smiles):
    """Element counts excluding H (the metric's cheap gate), or None if unparseable."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym != 'H':
            counts[sym] = counts.get(sym, 0) + 1
    return tuple(sorted(counts.items()))


def dedupe_candidates(candidates, max_rank=25):
    """candidates: iterable of (smiles, score), higher score = better.

    Returns up to max_rank (smiles, score, ik14) sorted best first, keeping one entry per
    metric-canonical InChIKey14 and dropping invalid SMILES. Two spellings or tautomers of one
    molecule would otherwise take two of the 25 slots, because rank is position in the list.
    """
    best = {}
    for smi, sc in candidates:
        ik = canonicalize_smiles_to_ik14(smi)
        if ik is None:
            continue
        if ik not in best or sc > best[ik][1]:
            best[ik] = (smi, sc, ik)
    ranked = sorted(best.values(), key=lambda t: t[1], reverse=True)
    return ranked[:max_rank]


def to_submission_string(ranked, max_rank=25):
    """ranked: output of dedupe_candidates -> 'smi1;smi2;...' as the submission expects."""
    return ';'.join(smi for smi, _, _ in ranked[:max_rank])


def reciprocal_rank(guesses, answer_smiles):
    """Score one molecule. answer_smiles may hold several accepted answers joined by ','."""
    candidates = [s for s in answer_smiles.split(',') if s]
    acceptable = {k for k in map(canonicalize_smiles_to_ik14, candidates) if k}
    accept_keys = {k for k in map(heavy_atom_composition, candidates) if k}
    if not acceptable or not accept_keys:
        raise ValueError(f'answer could not be parsed: {answer_smiles!r}')
    for rank, guess in enumerate(guesses, 1):
        if heavy_atom_composition(guess) not in accept_keys:
            continue
        if canonicalize_smiles_to_ik14(guess) in acceptable:
            return 1.0 / rank
    return 0.0


def mrr_at_k(guesses_by_molecule, answers, max_rank=25):
    """guesses_by_molecule: {molecule_id: [smiles, ...]}; answers: {molecule_id: smiles}."""
    scores = []
    for mol_id, ans in answers.items():
        g = list(guesses_by_molecule.get(mol_id, []))
        if len(g) > max_rank:
            raise ValueError(f'{mol_id}: {len(g)} guesses > {max_rank}')
        scores.append(reciprocal_rank(g, ans))
    return float(np.mean(scores))
