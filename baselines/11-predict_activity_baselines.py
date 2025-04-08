# conda activate chemprop
import os
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import RDLogger
import sys

CURRENT_DIR = Path(__file__).resolve().parent
os.chdir(CURRENT_DIR / "../chemprop")

import chemprop

# Suppress warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

polygon_path = Path(__file__).parent / "../../polygon/results/"
mtmol_gpt_path = Path(__file__).parent / "../../MTMol-GPT/results/"
deeplig_path = Path(__file__).parent / "../../DeepLig/results/"
diseases = {
    "schizophrenia": ["_5HT2A", "D2R"],
    "alzheimer": ["AChE", "MAOB"],
    "parkinson": ["D2R", "D3R"]
}

for disease in diseases:
    print(f"Processing {disease}")

    # POLYGON
    smiles = pd.read_csv(polygon_path / f"{disease}_sample.smi", header=None).iloc[:, 0].tolist()
    smiles_input = [[s] for s in smiles]
    models = diseases[disease]
    activity_df = pd.DataFrame({'SMILES': smiles})

    for model in models:
        if f"{model}_pXC50" in activity_df.columns:
            print(f"Skipping {model} - pXC50")
            continue

        print(f"Running inference for {model} - pXC50")
        arguments = [
            '--test_path', '/dev/null',
            '--preds_path', '/dev/null',
            '--checkpoint_dir', f'../models_chemprop/{model}-pXC50-checkpoint'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        preds = chemprop.train.make_predictions(args=args, smiles=smiles_input)

        activity_df[f"{model}_pXC50"] = np.array(preds).flatten()

    # Replace 'Invalid SMILES' with NaN for Activity and Inhibition
    activity_df = activity_df.replace('Invalid SMILES', np.nan)
    # Convert all columns except 'SMILES' to float
    activity_df = activity_df.astype(
        {col: 'float' for col in activity_df.columns if col != 'SMILES'}
    )

    combination = "_".join(models)

    activity_df[f"{combination}_pXC50"] = activity_df[
        [f"{model}_pXC50" for model in models]
    ].mean(axis=1, skipna=True)

    # Save the updated dataframe to new path
    new_path = polygon_path / f"{disease}_sample_activity.csv"
    activity_df.to_csv(new_path, index=False)

    # MTMol-GPT
    smiles = pd.read_csv(
        mtmol_gpt_path / f"{disease}/finetune_gail_num10000_e995.txt", header=None
    ).iloc[:, 0].tolist()
    smiles_input = [[s] for s in smiles]
    models = diseases[disease]
    activity_df = pd.DataFrame({'SMILES': smiles})

    for model in models:
        if f"{model}_pXC50" in activity_df.columns:
            print(f"Skipping {model} - pXC50")
            continue

        print(f"Running inference for {model} - pXC50")
        arguments = [
            '--test_path', '/dev/null',
            '--preds_path', '/dev/null',
            '--checkpoint_dir', f'../models_chemprop/{model}-pXC50-checkpoint'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        preds = chemprop.train.make_predictions(args=args, smiles=smiles_input)

        activity_df[f"{model}_pXC50"] = np.array(preds).flatten()

    # Replace 'Invalid SMILES' with NaN for Activity and Inhibition
    activity_df = activity_df.replace('Invalid SMILES', np.nan)
    # Convert all columns except 'SMILES' to float
    activity_df = activity_df.astype(
        {col: 'float' for col in activity_df.columns if col != 'SMILES'}
    )

    combination = "_".join(models)

    activity_df[f"{combination}_pXC50"] = activity_df[
        [f"{model}_pXC50" for model in models]
    ].mean(axis=1, skipna=True)

    # Save the updated dataframe to new path
    new_path = mtmol_gpt_path / f"{disease}_sample_activity.csv"
    activity_df.to_csv(new_path, index=False)

    # DeepLig
    smiles = pd.read_csv(deeplig_path / f"{disease}_SMILES.smi", header=None).iloc[:, 0].tolist()
    smiles_input = [[s] for s in smiles]
    models = diseases[disease]
    activity_df = pd.DataFrame({'SMILES': smiles})

    for model in models:
        if f"{model}_pXC50" in activity_df.columns:
            print(f"Skipping {model} - pXC50")
            continue

        print(f"Running inference for {model} - pXC50")
        arguments = [
            '--test_path', '/dev/null',
            '--preds_path', '/dev/null',
            '--checkpoint_dir', f'../models_chemprop/{model}-pXC50-checkpoint'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        preds = chemprop.train.make_predictions(args=args, smiles=smiles_input)

        activity_df[f"{model}_pXC50"] = np.array(preds).flatten()

    # Replace 'Invalid SMILES' with NaN for Activity and Inhibition
    activity_df = activity_df.replace('Invalid SMILES', np.nan)
    # Convert all columns except 'SMILES' to float
    activity_df = activity_df.astype(
        {col: 'float' for col in activity_df.columns if col != 'SMILES'}
    )

    combination = "_".join(models)

    activity_df[f"{combination}_pXC50"] = activity_df[
        [f"{model}_pXC50" for model in models]
    ].mean(axis=1, skipna=True)

    # Save the updated dataframe to new path
    new_path = deeplig_path / f"{disease}_sample_activity.csv"
    activity_df.to_csv(new_path, index=False)
