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

EPOCH = 25 if len(sys.argv) < 2 else sys.argv[1]
PREDICTED_ACTIVITY_PATH = CURRENT_DIR / ".." / "generated_molecules" / f"{EPOCH}-epoch" / "predicted_activity"

# Multi-target prediction
multitarget_combination = {
    "AChE_MAOB": {
        "generated_molecules": ["AChE", "MAOB", "AChE_MAOB_SUM", "AChE_MAOB_MEAN", "AChE_MAOB_MAX"],
        "models": ["AChE", "MAOB"]
    },
    "D2R_D3R": {
        "generated_molecules": ["D2R", "D3R", "D2R_D3R_SUM", "D2R_D3R_MEAN", "D2R_D3R_MAX"],
        "models": ["D2R", "D3R"]
    },
    "D2R__5HT2A": {
        "generated_molecules": ["D2R", "_5HT2A", "D2R__5HT2A_SUM", "D2R__5HT2A_MEAN", "D2R__5HT2A_MAX"],
        "models": ["D2R", "_5HT2A"]
    },
}

for combination in multitarget_combination:
    print(f"Processing {combination} combination")

    mols_activity = multitarget_combination[combination]["generated_molecules"]
    mols_activity_paths = [PREDICTED_ACTIVITY_PATH / f"{m}.csv" for m in mols_activity]

    models = multitarget_combination[combination]["models"]

    for path in mols_activity_paths:
        print(f"Processing {path.resolve()}")

        activity_df = pd.read_csv(path)

        smiles = activity_df['SMILES'].tolist()

        for model in models:
            if f"{model}_pXC50" in activity_df.columns:
                print(f"Skipping {model} - pXC50")
                continue

            print(f"Running inference for {model} - pXC50")
            preds = predict_activity_chemprop(smiles, model)
            activity_df[f"{model}_pXC50"] = np.array(preds)

        # Replace 'Invalid SMILES' with NaN for Activity and Inhibition
        activity_df = activity_df.replace('Invalid SMILES', np.nan)
        # Convert all columns except 'SMILES' to float
        activity_df = activity_df.astype(
            {col: 'float' for col in activity_df.columns if col != 'SMILES'}
        )


        activity_df[f"{combination}_pXC50"] = activity_df[
            [f"{model}_pXC50" for model in models]
        ].mean(axis=1, skipna=True)
    
        # Save the updated dataframe to new path
        new_path = PREDICTED_ACTIVITY_PATH / f"{path.stem}.csv"
        activity_df.to_csv(new_path, index=False)

# Unconditional
print("Processing Unconditional")

mols_activity_paths = PREDICTED_ACTIVITY_PATH / "Unconditional.csv"
unconditional_df = pd.read_csv(mols_activity_paths)

for combination in multitarget_combination:
    print(f"Processing {combination} combination")
    models = multitarget_combination[combination]["models"]

    # Replace 'Invalid SMILES' with NaN for Activity and Inhibition
    unconditional_df = unconditional_df.replace('Invalid SMILES', np.nan)
    # Convert all columns except 'SMILES' to float
    unconditional_df = unconditional_df.astype(
        {col: 'float' for col in unconditional_df.columns if col != 'SMILES'}
    )

    unconditional_df[f"{combination}_pXC50"] = unconditional_df[
        [f"{model}_pXC50" for model in models]
    ].mean(axis=1, skipna=True)

    # Save the updated dataframe to new path
    new_path = PREDICTED_ACTIVITY_PATH / f"Unconditional.csv"
    unconditional_df.to_csv(new_path, index=False)
