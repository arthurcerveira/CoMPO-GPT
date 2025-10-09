# %%
import pandas as pd
from pathlib import Path
import random
import numpy as np
from sklearn.model_selection import train_test_split

# Set seed
random.seed(1907)
np.random.seed(1907)

current_dir = Path(__file__).resolve().parent

assays_dir = current_dir / ".." / ".." / ".." / "data" / "Assays-pXC50"
assays_paths = list(assays_dir.glob("*.csv"))
output_path = current_dir / "./DLGN"

for assay_path in assays_paths:
    assay_name = assay_path.stem
    assay_df = pd.read_csv(assay_path)
    assay_df["pXC50"] = pd.to_numeric(assay_df["pXC50"], errors="coerce")
    assay_df = assay_df.dropna(how="any")
    active_smiles = assay_df[assay_df["pXC50"] >= 6.0]["SMILES"]

    # Split data in train/valid
    train, valid = train_test_split(active_smiles, test_size=0.2, random_state=1907)

    # Save data
    train_path = output_path / f"{assay_name}_train.txt"
    valid_path = output_path / f"{assay_name}_valid.txt"
    train.to_csv(train_path, index=False, header=False)
    valid.to_csv(valid_path, index=False, header=False)
    print(f"Saved {len(train)} train and {len(valid)} valid samples for {assay_name}")
    train.to_csv(train_path, index=False, header=False)
    valid.to_csv(valid_path, index=False, header=False)
    print(f"Saved {len(train)} train and {len(valid)} valid samples for {assay_name}")


