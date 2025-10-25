# %%
import pandas as pd
from pathlib import Path
import random
import numpy as np
import json
from sklearn.model_selection import train_test_split

# Set seed
random.seed(1907)
np.random.seed(1907)

current_dir = Path(__file__).resolve().parent

assays_dir = current_dir / ".." / ".." / ".." / "data" / "Assays-pXC50"
assays_paths = list(assays_dir.glob("*.csv"))
output_path = current_dir / "./DLGN"

# for assay_path in assays_paths:
#     assay_name = assay_path.stem
#     assay_df = pd.read_csv(assay_path)
#     assay_df["pXC50"] = pd.to_numeric(assay_df["pXC50"], errors="coerce")
#     assay_df = assay_df.dropna(how="any")
#     active_smiles = assay_df[assay_df["pXC50"] >= 6.0]["SMILES"]

#     # Split data in train/valid
#     train, valid = train_test_split(active_smiles, test_size=0.2, random_state=1907)

#     # Save data
#     train_path = output_path / f"{assay_name}_train.txt"
#     valid_path = output_path / f"{assay_name}_valid.txt"
#     train.to_csv(train_path, index=False, header=False)
#     valid.to_csv(valid_path, index=False, header=False)
#     print(f"Saved {len(train)} train and {len(valid)} valid samples for {assay_name}")

# Load MPO data
print("\nProcessing MPO data...")
data_dir = current_dir / ".." / ".." / ".." / "data"

# Define valid tokens (from the model's vocabulary)
valid_tokens = set([' ', '<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
                   '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
                   '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r'])

def is_valid_smiles(smiles):
    """Check if SMILES contains only valid tokens"""
    return all(char in valid_tokens for char in smiles)

# Load target conditions mapping
with open(data_dir / "target_conditions_to_index_mpo.json", "r") as f:
    target_mapping = json.load(f)

# Identify target indices
target_indices = {
    "BBB": target_mapping["gene_to_index"]["BBB"],
    "CNSMPO": target_mapping["gene_to_index"]["CNSMPO"], 
    "SAScore": target_mapping["gene_to_index"]["SAScore"]
}

print(f"Target indices: {target_indices}")

# Load active compounds
mpo_file = data_dir / "active_compounds_mpo.smi"
mpo_data = []
invalid_smiles_count = 0

with open(mpo_file, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            smiles, target_idx = parts[0], int(parts[1])
            if is_valid_smiles(smiles):
                mpo_data.append((smiles, target_idx))
            else:
                invalid_smiles_count += 1

print(f"Loaded {len(mpo_data)} MPO compounds")
print(f"Filtered out {invalid_smiles_count} compounds with invalid tokens")

# Process each target
for target_name, target_idx in target_indices.items():
    print(f"\nProcessing {target_name} (index {target_idx})...")
    
    # Filter molecules for this target
    target_molecules = [smiles for smiles, idx in mpo_data if idx == target_idx]
    
    if len(target_molecules) == 0:
        print(f"No molecules found for {target_name}")
        continue
        
    print(f"Found {len(target_molecules)} valid molecules for {target_name}")
    
    # Split data in train/valid
    train, valid = train_test_split(target_molecules, test_size=0.2, random_state=1907)
    
    # Save data
    train_path = output_path / f"{target_name}_train.txt"
    valid_path = output_path / f"{target_name}_valid.txt"
    
    # Save as text files (one SMILES per line)
    with open(train_path, "w") as f:
        for smiles in train:
            f.write(smiles + "\n")
    
    with open(valid_path, "w") as f:
        for smiles in valid:
            f.write(smiles + "\n")
    
    print(f"Saved {len(train)} train and {len(valid)} valid samples for {target_name}")


