import os
import pandas as pd
from sklearn.model_selection import train_test_split
from metric import score
import numpy as np

TRAIN_PATH = 'data/train.parquet'
NEEDED = ['ingest_lib', 'inchikey14', 'normalized_smiles', 'precursor_mz',
          'ms2_mzs', 'ms2_normalized_intensities']
os.makedirs('outputs', exist_ok=True)

df = pd.read_parquet(TRAIN_PATH, columns=NEEDED,
                     filters=[('ingest_lib', '!=', 'enveda-180')])

# same split as the tutorial, so the whole team shares it
all_structs = np.asarray(df.inchikey14.unique(), dtype=object)
_, val_structs = train_test_split(all_structs, test_size=500, random_state=0)

# up to 3 spectra per molecule, like the test (~2.8 per molecule)
val = df[df.inchikey14.isin(val_structs)]
val = (val.sample(frac=1, random_state=0)
          .groupby('inchikey14').head(3).reset_index(drop=True))
val['molecule_id'] = 'v_' + val.inchikey14

solution = (val.groupby('molecule_id').normalized_smiles
            .agg(lambda s: ','.join(sorted(set(s)))).reset_index()
            .rename(columns={'normalized_smiles': 'smiles'}))
solution['Usage'] = 'Public'
print(len(val), 'spectra,', len(solution), 'molecules')

pd.Series(val_structs).to_csv('outputs/val_structs.csv', index=False, header=['inchikey14'])
val.to_parquet('outputs/val_spectra.parquet')
solution.to_csv('outputs/val_solution.csv', index=False)

# sanity checks for the harness
ids = solution.molecule_id.tolist()
first = [s.split(',')[0] for s in solution.smiles]
def sub(lists):
    return pd.DataFrame({'molecule_id': ids, 'smiles': [';'.join(l) for l in lists]})
print('perfect (expect 1.0):', score(solution, sub([[s] for s in first]), 'molecule_id'))
print('rank 5  (expect 0.2):', score(solution, sub([['CCO']*4 + [s] for s in first]), 'molecule_id'))
print('constant (expect ~0):', score(solution, sub([['CCO'] for _ in first]), 'molecule_id'))