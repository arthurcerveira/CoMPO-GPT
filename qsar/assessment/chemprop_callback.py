from pathlib import Path
import json
import sys

current_dir = Path(__file__).parent
chemprop_path = current_dir / '..' / "baselines" / "chemprop"
sys.path.append(str(chemprop_path))

import numpy as np
from lightning import pytorch as pl
import torch
import torch.nn.functional as F
from torch import Tensor
torch.set_float32_matmul_precision("high")

# Suppress warnings
# Comment out: <env>/lib/python3.12/site-packages/lightning/pytorch/accelerators/cuda.py
from pytorch_lightning.utilities import rank_zero
rank_zero.rank_zero_info = lambda *a, **k: None
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from chemprop import data, featurizers, models, nn


checkpoints_dir = current_dir / ".." / "checkpoints"
data_dir = current_dir / ".." / "data"


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


def run_mpnn_on_smiles(smiles_input, mpnn):
    # Create list to track valid/invalid SMILES
    valid_smiles_idx = []
    test_data = []
    
    # Try to create MoleculeDatapoint for each SMILES, track valid ones
    for i, smi in enumerate(smiles_input):
        try:
            datapoint = data.MoleculeDatapoint.from_smi(smi)
            test_data.append(datapoint)
            valid_smiles_idx.append(i)
        except:
            continue

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    test_dset = data.MoleculeDataset(test_data, featurizer=featurizer)
    test_loader = data.build_dataloader(test_dset, shuffle=False, num_workers=4, batch_size=4096)

    # Suppress PyTorch Tensor Core warning
    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=False,
            enable_progress_bar=False,
            accelerator="cuda", 
            devices=1,
            inference_mode=True,
            enable_checkpointing=False,
            enable_model_summary=False,
            log_every_n_steps=0
        )
        test_preds = trainer.predict(mpnn, test_loader)
        test_preds = np.concatenate(test_preds, axis=0)  # Concatenate batches

    # Create array of np.nan values with same length as input
    full_preds = np.array([np.nan] * len(smiles_input))
    
    # Check if model is single or multi-target
    tasks = test_preds.shape[1]
    if tasks == 1:
        # Single target: fill in predictions for valid SMILES
        full_preds[valid_smiles_idx] = test_preds[:, 0]
        return full_preds
        
    # Multi-target: create 2D array of None values
    full_preds = np.array([[np.nan] * len(smiles_input) for _ in range(tasks)])
    for i in range(tasks):
        full_preds[i][valid_smiles_idx] = test_preds[:, i]
    return full_preds


def chemprop_single_target_callback(test_dataset, targets, trained_on="TVT", verbose=False, dropna=True):
    target_predictions = dict()

    for target in targets:
        if dropna:
            smiles_input = test_dataset.dropna(subset=[target])["SMILES"].tolist()
        else:  # Used for knowledge distillation
            smiles_input = test_dataset["SMILES"].tolist()

        if verbose:
            print(f"Running ST-Chemprop on {len(smiles_input)} SMILES for target {target}...")

        checkpoint_path = checkpoints_dir / "target-specific" / trained_on / target / "last.ckpt"
        mpnn = models.MPNN.load_from_checkpoint(checkpoint_path)
        predictions = run_mpnn_on_smiles(smiles_input, mpnn)
        target_predictions[target] = predictions

    return target_predictions


def chemprop_multi_target_callback(test_dataset, targets, trained_on="TVT", verbose=False):
    if verbose:
        print(f"Running MT-Chemprop on {len(test_dataset)} SMILES for {len(targets)} targets...")
    
    target_predictions = dict()
    with open(checkpoints_dir / "multi-target" / trained_on / "MT-ALL" / "index_to_target.json", "r") as f:
        index_to_target = json.load(f)
    
    smiles_input = test_dataset["SMILES"].tolist()
    mpnn = models.MPNN.load_from_checkpoint(checkpoints_dir / "multi-target" / trained_on / "MT-ALL" / "last.ckpt")
    predictions = run_mpnn_on_smiles(smiles_input, mpnn)

    target_to_index = {v: k for k, v in index_to_target.items()}
    for target in targets:
        target_mask = test_dataset[target].notna()
        target_idx = int(target_to_index[target])
        target_predictions[target] = predictions[target_idx][target_mask]

    return target_predictions


def chemprop_clustered_multi_target_callback(test_dataset, targets, trained_on="TVT", verbose=False):
    target_predictions = dict()

    label = trained_on.split('-')[0]
    with open(data_dir / f"target_clusters_correlation_{label}.json", "r") as f:
        target_clusters = json.load(f)

    for cluster_idx in target_clusters["cluster_to_targets"]:
        clustered_targets = target_clusters["cluster_to_targets"][cluster_idx]

        checkpoint_path = checkpoints_dir / "clustered-multi-target" / trained_on / f"cluster-{cluster_idx}" / "last.ckpt"
        mpnn = models.MPNN.load_from_checkpoint(checkpoint_path)
        
        target_dataset = test_dataset.dropna(subset=clustered_targets, how="all")

        if verbose:
            print(
                f"Running clustered MT-Chemprop on {len(target_dataset)} SMILES "
                f"for {len(clustered_targets)} targets (cluster {cluster_idx})..."
            )

        smiles_input = target_dataset["SMILES"].tolist()
        predictions = run_mpnn_on_smiles(smiles_input, mpnn)
        if predictions.ndim == 1:
            predictions = np.array([predictions])  # Convert to 2D array

        for target, preds in zip(clustered_targets, predictions):
            target_mask = target_dataset[target].notna()
            target_predictions[target] = preds[target_mask]

    return target_predictions


with open(data_dir / f"target_clusters_correlation_TVT.json", "r") as f:
    target_clusters_tvt = json.load(f)


def add_masked_mse_to_modules():
    """
    Avoids the error: 
        AttributeError: Can't get attribute 'MaskedMSE' on <module 
        '__main__' (<class '_frozen_importlib.BuiltinImporter'>)>
    when importing the module in other scripts.
    """
    import sys
    # Add the MaskedMSE class to the __main__ module so PyTorch can find it
    if '__main__' not in sys.modules:
        import types
        sys.modules['__main__'] = types.ModuleType('__main__')
    
    # Add the MaskedMSE class to __main__ module
    sys.modules['__main__'].MaskedMSE = MaskedMSE


def load_model_from_target(target):
    add_masked_mse_to_modules()

    if target == "BBB":
        return models.MPNN.load_from_checkpoint(
            checkpoints_dir / "target-specific" / "TVT" / "BBB" / "last.ckpt",
            map_location="cuda",
        ).eval()

    cluster = target_clusters_tvt["targets_to_cluster"][target]
    return models.MPNN.load_from_checkpoint(
        checkpoints_dir / "clustered-multi-target" / "TVT-KD" / f"cluster-{cluster}" / "last.ckpt",
        map_location="cuda",
    ).eval()


def predict_activity_chemprop(smiles, target, model=None):
    """
    Predict activity for a target using a Chemprop model.
    Always use the clustered MT-Chemprop model trained on TVT-KD.
    First, select the model based on the target.
    Then, predict the SMILES activity for the target.
    If the model is not provided, it will be loaded from the checkpoint.
    """
    add_masked_mse_to_modules()

    cluster = target_clusters_tvt["targets_to_cluster"][target]
    target_idx_in_cluster = target_clusters_tvt["cluster_to_targets"][str(cluster)].index(target)

    if model is None:
        model = load_model_from_target(target)

    predictions = run_mpnn_on_smiles(smiles, model)
    # If there is only one target in the cluster, we must reshape it to (1, # molecules)
    if predictions.ndim == 1:
        predictions = np.array([predictions])

    return predictions[target_idx_in_cluster]


def predict_bbb_chemprop(smiles, model=None):
    """
    Predict the molecule's ability to cross the BBB using a Chemprop model.
    Checkpoint: checkpoints_dir / "target-specific" / "TVT" / "BBB" / "last.ckpt"
    If the model is not provided, it will be loaded from the checkpoint.
    """
    add_masked_mse_to_modules()

    if model is None:
        model = load_model_from_target("BBB")

    predictions = run_mpnn_on_smiles(smiles, model)
    return predictions  # Predictions will have shape (# of molecules,)
