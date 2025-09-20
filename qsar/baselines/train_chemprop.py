import os
from pathlib import Path
import json
import sys

current_file_dir = Path(__file__).resolve().parent
chemprop_path = current_file_dir / "chemprop"
sys.path.append(str(chemprop_path))

import numpy as np
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from chemprop import data, featurizers, models, nn, conf

DATASET = "TVT"
RETRAIN = False

checkpoints_dir = current_file_dir / ".." / "checkpoints"
results_dir = current_file_dir / ".." / "results"
data_dir = current_file_dir / ".." / "data"

# Reference: https://chemprop.readthedocs.io/en/latest/training.html
input_path = data_dir / "bbb_for_training.csv"  # created from MPO preprocessing notebook
activities_df = pd.read_csv(input_path)
num_workers = 0 # number of workers for dataloader. 0 means using main process for data loading
smiles_column = 'SMILES' # name of the column containing SMILES strings
targets = ["activity"]
total_targets = len(targets)


@nn.metrics.LossFunctionRegistry.register("masked_mse")
@nn.metrics.MetricRegistry.register("masked_mse")
class MaskedMSE(nn.metrics.ChempropMetric):
    def update(
        self,
        preds: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
        lt_mask: Tensor | None = None,
        gt_mask: Tensor | None = None,
    ) -> None:
        """Update total loss by considering only valid targets per task."""
        # Auto-set mask where targets are NaN
        mask = ~torch.isnan(targets) if mask is None else mask  
        targets = torch.where(mask, targets, torch.zeros_like(targets))  # Replace NaN with 0 (ignored due to mask)

        weights = torch.ones_like(targets, dtype=torch.float) if weights is None else weights
        lt_mask = torch.zeros_like(targets, dtype=torch.bool) if lt_mask is None else lt_mask
        gt_mask = torch.zeros_like(targets, dtype=torch.bool) if gt_mask is None else gt_mask

        # Compute loss, apply weights and mask
        L = self._calc_unreduced_loss(preds, targets, mask)  
        L = L * weights * self.task_weights * mask  

        # Aggregate loss per task
        valid_counts = mask.sum(dim=0)  # Count valid values per task
        per_task_loss = L.sum(dim=0) / valid_counts.clamp(min=1)  # Avoid division by zero

        # Sum over tasks
        self.total_loss += per_task_loss.sum()
        self.num_samples += valid_counts.sum()

    def compute(self):
        return self.total_loss / self.num_samples if self.num_samples > 0 else torch.tensor(0.0)

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        """Compute the element-wise loss while ignoring missing values based on mask."""
        loss = F.mse_loss(preds, targets, reduction="none")  # Compute MSE loss
        return loss * mask  # Zero out loss for missing targets


def data_pre_processing(activity_df, targets_columns, smiles_column, num_workers):
    df_input = activities_df[[smiles_column] + targets_columns + ["split"]].copy()
    
    # Drop rows with missing values in all targets columns
    df_input = df_input.dropna(subset=targets_columns, how="all")
    
    smis = df_input.loc[:, smiles_column].values
    ys = df_input.loc[:, targets_columns].values
    splits = df_input.loc[:, "split"].values
    print(f"Number of data points: {len(smis):,}")
    
    # For binary classification, ensure targets are 0 or 1
    # Assuming the target column contains binary values (0/1) or needs to be converted
    if len(targets_columns) == 1:
        # Convert to binary if needed (assuming values are already 0/1 or can be thresholded)
        # Check if values are already binary (0/1)
        unique_values = np.unique(ys)
        if not np.array_equal(unique_values, [0, 1]) and not np.array_equal(unique_values, [0.0, 1.0]):
            print(f"Converting continuous values to binary using threshold 0.5. Unique values: {unique_values}")
            ys = (ys > 0.5).astype(float)  # Threshold at 0.5 if needed, adjust as necessary
        else:
            print(f"Targets are already binary. Unique values: {unique_values}")
            ys = ys.astype(float)  # Ensure float type
        
        # Print class distribution
        class_counts = np.bincount(ys.astype(int).flatten())
        print(f"Class distribution: {dict(zip(range(len(class_counts)), class_counts))}")
        print(f"Class proportions: {class_counts / class_counts.sum()}")
    
    all_data = [
        data.MoleculeDatapoint.from_smi(smi, y) for smi, y in 
        tqdm(zip(smis, ys), total=len(smis), desc="Processing molecules")
    ]

    # Get indices for train, val, and test from splits column
    train_indices, val_indices, test_indices = (
        np.array(np.where(splits == "train")), 
        np.array(np.where(splits == "val")),
        np.array(np.where(splits == "test"))
    )

    # If there is no validation set, set the last 10% of the training set as the validation set
    if len(val_indices[0]) == 0:
        val_size = int(len(train_indices[0]) * 0.1)
        val_indices = np.array([train_indices[0][-val_size:]])
        train_indices = np.array([train_indices[0][:-val_size]])

    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    # For binary classification, no target normalization is needed
    train_dset = data.MoleculeDataset(train_data[0], featurizer)
    val_dset = data.MoleculeDataset(val_data[0], featurizer)
    test_dset = data.MoleculeDataset(test_data[0], featurizer)

    train_loader = data.build_dataloader(train_dset, num_workers=num_workers, batch_size=1024)
    val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False, batch_size=1024)
    test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False, batch_size=1024)

    return train_loader, val_loader, test_loader


def train_single_target(target, t_idx):
    print(f"({t_idx}/{total_targets}) Training model for {target}")
    
    # BBB hardcoded for this training sctipt
    checkpoint_path = checkpoints_dir / "target-specific" / DATASET / "BBB"
    if not RETRAIN and os.path.exists(checkpoint_path / "last.ckpt"):
        print(f"Model for {target} already trained. Skipping...")
        return

    train_loader, val_loader, test_loader = data_pre_processing(
        activities_df, [target], smiles_column, num_workers
    )

    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    # No output transform needed for binary classification
    ffn = nn.BinaryClassificationFFN()
    batch_norm = True

    # Use appropriate metrics for binary classification
    metric_list = [nn.metrics.BCELoss(), nn.metrics.BinaryAUROC(), nn.metrics.BinaryAccuracy()] # Only the first metric is used for training and early stopping
    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

    # Configure model checkpointing
    checkpointing = ModelCheckpoint(
        checkpoint_path,  # Directory where model checkpoints will be saved
        "best-{epoch}-{val_loss:.2f}",  # Filename format for checkpoints, including epoch and validation loss
        "val_loss",  # Metric used to select the best checkpoint (based on validation loss)
        mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
        save_last=True,  # Always save the most recent checkpoint, even if it's not the best
        enable_version_counter=False
    )

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=20, # number of epochs to train for
        callbacks=[checkpointing], # Use the configured checkpoint callback
    )

    print(f"Saving checkpoints to {checkpoint_path.resolve()}")
    trainer.fit(mpnn, train_loader, val_loader)
    results = trainer.test(dataloaders=test_loader)
    return results


def train_multi_target(targets):
    checkpoint_path = checkpoints_dir / "multi-target" / DATASET / "MT-ALL"
    if not RETRAIN and os.path.exists(checkpoint_path / "last.ckpt"):
        print(f"Multi-target model already trained. Skipping...")
        return

    train_loader, val_loader, test_loader = data_pre_processing(
        activities_df, targets, smiles_column, num_workers
    )

    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    # No output transform needed for binary classification

    # Adjusted FFN to handle multiple targets for binary classification
    ffn = nn.BinaryClassificationFFN(n_tasks=len(targets))

    batch_norm = True
    metric_list = [nn.metrics.BCELoss()]  # Only the first metric is used for training and early stopping
    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

    # Configure model checkpointing
    checkpointing = ModelCheckpoint(
        checkpoint_path,  # Directory where model checkpoints will be saved
        "best-{epoch}-{val_loss:.2f}",  # Filename format for checkpoints, including epoch and validation loss
        "val_loss",  # Metric used to select the best checkpoint (based on validation loss)
        mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
        save_last=True,  # Always save the most recent checkpoint, even if it's not the best
        enable_version_counter=False
    )

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=20, # number of epochs to train for
        callbacks=[checkpointing], # Use the configured checkpoint callback
    )

    trainer.fit(mpnn, train_loader, val_loader)
    results = trainer.test(dataloaders=test_loader)

    # Save index-to-target mapping to checkpoint directory for later use
    index_to_target = {i: target for i, target in enumerate(targets)}
    with open(checkpoint_path / "index_to_target.json", "w") as f:
        json.dump(index_to_target, f, indent=4, ensure_ascii=False)

    return results


def train_clustered_multi_target(clustered_targets, cluster_idx):
    print(f"Training clustered multi-target model for {len(clustered_targets)} targets (cluster {cluster_idx})")

    checkpoint_path = checkpoints_dir / "clustered-multi-target" / DATASET / f"cluster-{cluster_idx}"
    if not RETRAIN and os.path.exists(checkpoint_path / "last.ckpt"):
        print(f"Clustered multi-target model for cluster {cluster_idx} already trained. Skipping...")
        return

    train_loader, val_loader, test_loader = data_pre_processing(
        activities_df, clustered_targets, smiles_column, num_workers
    )
    
    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    # No output transform needed for binary classification

    ffn = nn.BinaryClassificationFFN(n_tasks=len(clustered_targets))
    
    batch_norm = True
    metric_list = [nn.metrics.BCELoss()]  # Only the first metric is used for training and early stopping
    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

    # Configure model checkpointing
    checkpointing = ModelCheckpoint(
        checkpoint_path,  # Directory where model checkpoints will be saved
        "best-{epoch}-{val_loss:.2f}",  # Filename format for checkpoints, including epoch and validation loss
        "val_loss",  # Metric used to select the best checkpoint (based on validation loss)
        mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
        save_last=True,  # Always save the most recent checkpoint, even if it's not the best
        enable_version_counter=False
    )

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=20, # number of epochs to train for
        callbacks=[checkpointing], # Use the configured checkpoint callback
    )

    trainer.fit(mpnn, train_loader, val_loader)
    results = trainer.test(dataloaders=test_loader)

    return results


if __name__ == "__main__":
    # Train for BBB dataset
    checkpoint_path = checkpoints_dir / "target-specific" / DATASET
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    print("Training target-specific models.")
    all_results = dict()

    for t_idx, target in enumerate(targets):
        results = train_single_target(target, t_idx)
        all_results[target] = results

    print("Training complete.")
    results_path = results_dir / "target-specific" / "chemprop-train.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print("Results saved.")
