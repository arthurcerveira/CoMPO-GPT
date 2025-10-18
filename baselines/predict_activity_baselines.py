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

GENERATED_MOLS_PATH = CURRENT_DIR / ".." / "generated_molecules"

# Define baseline paths and their corresponding file patterns
baseline_configs = {
    "POLYGON": {
        "path": GENERATED_MOLS_PATH / "POLYGON/",
    },
    "MTMol-GPT": {
        "path": GENERATED_MOLS_PATH / "MTMol-GPT/",
    },
    "DeepLig": {
        "path": GENERATED_MOLS_PATH / "DeepLig/",
    }
}

diseases = {
    "schizophrenia": ["_5HT2A", "D2R"],
    "alzheimer": ["AChE", "MAOB"],
    "parkinson": ["D2R", "D3R"]
}

def process_baseline_activity(baseline_name, baseline_config, disease, models):
    """
    Process activity prediction for a single baseline and disease combination.
    Similar to the main script's approach but adapted for baseline file structures.
    """
    print(f"Processing {baseline_name} for {disease}")
    
    # Construct file path
    file_path = baseline_config["path"] / f"{disease}.csv"
    
    # Check if file exists
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    
    # Read SMILES
    try:
        smiles = pd.read_csv(file_path, header=None).iloc[:, 0].tolist()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Create predictions dataframe
    predictions = pd.DataFrame({'SMILES': smiles})
    
    # Predict activity for each model
    for model in models:
        print(f"Running inference for {model} - pXC50")
        try:
            preds = predict_activity_chemprop(smiles, model)
            predictions[f"{model}_pXC50"] = np.array(preds)
        except Exception as e:
            print(f"Error predicting {model}: {e}")
            continue
    
    # Check if any predictions were made
    if len(predictions.columns) == 1:
        print(f"No activity predicted for {baseline_name} - {disease}")
        return
    
    # Create predicted_activity directory
    predicted_activity_path = baseline_config["path"] / "predicted_activity"
    predicted_activity_path.mkdir(exist_ok=True)
    
    # Save predictions
    output_file = predicted_activity_path / f"{disease}.csv"
    predictions.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")


# Process each disease and baseline combination
for disease in diseases:
    print(f"\n=== Processing {disease} ===")
    models = diseases[disease]
    
    for baseline_name, baseline_config in baseline_configs.items():
        process_baseline_activity(baseline_name, baseline_config, disease, models)
