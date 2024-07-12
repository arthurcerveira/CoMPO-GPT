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

EPOCH = 100 if len(sys.argv) < 2 else sys.argv[1]
GENERATED_MOLS_PATH = CURRENT_DIR / ".." / "generated_molecules" / f"{EPOCH}-epoch"

models = ['AChE', 'D2R', 'D3R', '_5HT2A', 'MAOB'] #, 'BBB']
generated_mols_paths = GENERATED_MOLS_PATH.glob('*.csv')

for path in generated_mols_paths:
    print(f"Processing {path.resolve()}")

    smiles = pd.read_csv(path)['SMILES'].tolist()
    smiles_input = [[s] for s in smiles]
    predictions = pd.DataFrame()
    predictions['SMILES'] = smiles

    for model in models:
        if model not in path.stem and path.stem != "Unconditional":
            continue

        print(f"Running inference for {model} - Activity")
        arguments = [
            '--test_path', '/dev/null',
            '--preds_path', '/dev/null',
            '--checkpoint_dir', f'../models_chemprop/{model}-checkpoint'
        ]
        args = chemprop.args.PredictArgs().parse_args(arguments)
        preds = chemprop.train.make_predictions(args=args, smiles=smiles_input)

        predictions[f"{model}_Activity"] = np.array(preds).flatten()

        # print(f"Running inference for {model} - pIC50")
        # arguments = [
        #     '--test_path', '/dev/null',
        #     '--preds_path', '/dev/null',
        #     '--checkpoint_dir', f'../models_chemprop/{model}-IC50-checkpoint'
        # ]
        # args = chemprop.args.PredictArgs().parse_args(arguments)
        # preds = chemprop.train.make_predictions(args=args, smiles=smiles_input)

        # predictions[f"{model}_pIC50"] = np.array(preds).flatten()

        print(f"Running inference for {model} - Inhibition")
        arguments = [
            '--test_path', '/dev/null',
            '--preds_path', '/dev/null',
            '--checkpoint_dir', f'../models_chemprop/{model}-inhibition-checkpoint'
        ]
        args = chemprop.args.PredictArgs().parse_args(arguments)
        preds = chemprop.train.make_predictions(args=args, smiles=smiles_input)

        predictions[f"{model}_inhibition"] = np.array(preds).flatten()


    predictions.to_csv(
        GENERATED_MOLS_PATH / "predicted_activity" / f"{path.stem}.csv", index=False
    )
