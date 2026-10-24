import pandas as pd
from metric import score

solution = pd.DataFrame({'molecule_id': val_ids, 'smiles': val_smiles, 'Usage': 'Public'})
submission = pd.DataFrame({'molecule_id': val_ids,
                           'smiles': [';'.join(c[:25]) for c in candidates]})
print(score(solution, submission, 'molecule_id', max_rank=25))