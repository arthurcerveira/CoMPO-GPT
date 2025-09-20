# conda activate chemprop
import os
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import RDLogger
import sys

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR / ".." / "qsar"))

from assessment.chemprop_callback import (
    predict_activity_chemprop, 
    predict_bbb_chemprop, 
    MaskedMSE
)

# Suppress warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

EPOCH = 100 if len(sys.argv) < 2 else sys.argv[1]
GENERATED_MOLS_PATH = CURRENT_DIR / ".." / "generated_molecules" / f"{EPOCH}-epoch"
PREDICTED_ACTIVITY_PATH = GENERATED_MOLS_PATH / "predicted_activity"
PREDICTED_ACTIVITY_PATH.mkdir(exist_ok=True)

models = ['AChE', 'D2R', 'D3R', '_5HT2A', 'MAOB'] #, 'BBB']
generated_mols_paths = GENERATED_MOLS_PATH.glob('*.csv')

for path in generated_mols_paths:
    print(f"Processing {path.resolve()}")

    smiles = pd.read_csv(path)['SMILES'].tolist()
    predictions = pd.DataFrame(data=smiles, columns=['SMILES'])

    for model in models:
        if model not in path.stem and path.stem != "Unconditional":
            continue

        preds = predict_activity_chemprop(smiles, model)
        predictions[f"{model}_pXC50"] = np.array(preds)  # .flatten()

    predictions.to_csv(
        GENERATED_MOLS_PATH / "predicted_activity" / f"{path.stem}.csv", index=False
    )
