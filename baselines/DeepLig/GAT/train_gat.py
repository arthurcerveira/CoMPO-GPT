import pandas as pd
from pathlib import Path
import sys

from moleculenet.utils import collate_molgraphs, load_model, predict
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from dgllife.utils import CanonicalAtomFeaturizer
from dgllife.utils import EarlyStopping, Meter, SMILESToBigraph
from dgllife.data import MoleculeCSVDataset
from moleculenet.utils import split_dataset
from moleculenet.utils import collate_molgraphs, load_model
from moleculenet.regression import run_a_train_epoch, run_an_eval_epoch


node_featurizer = CanonicalAtomFeaturizer()
edge_featurizer = None
smiles_to_g = SMILESToBigraph(add_self_loop=True, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer)
exp_config = {
  "model": "GAT",
  "device": "cuda",
  "alpha": None,
  "batch_size": 256,
  "dropout": None,
  "gnn_hidden_feats": None,
  "lr": 1e-3,
  "num_gnn_layers": 2,  # Only parameter specified in the paper
  "num_heads": None,
  "patience": 10,
  "predictor_hidden_feats": None,
  "residual": False,
  "weight_decay": 0,
  "n_tasks": 1,
  "in_node_feats": node_featurizer.feat_size()
}

args = {
    'split_ratio': '0.8,0.1,0.1',
    'split': 'random',
    'device': exp_config['device'],
    'metric': 'rmse',
    'result_path': 'checkpoints',
    'num_epochs': 50,  # Also specified in the paper
    'num_workers': 1,
    'edge_featurizer': edge_featurizer,
    'print_every': 50
}


if __name__ == '__main__':
    target = sys.argv[1]
    
    current_dir = Path(__file__).resolve().parent
    checkpoints_dir = current_dir / args['result_path']
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    assays_path = current_dir / ".." / ".." / ".." / "data" / "Assays-pXC50" / f"{target}.csv"
    assays = pd.read_csv(assays_path)
    assays = assays.dropna(subset=["SMILES", "pXC50"])
    dataset = MoleculeCSVDataset(
        assays, smiles_to_graph=smiles_to_g, 
        smiles_column='SMILES', task_names=['pXC50'],
        cache_file_path=f'{checkpoints_dir}/{target}_dataset.pth'
    )

    train_set, val_set, test_set = split_dataset(args=args, dataset=dataset)
    train_loader = DataLoader(dataset=train_set, batch_size=exp_config['batch_size'], shuffle=True,
                                collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    val_loader = DataLoader(dataset=val_set, batch_size=exp_config['batch_size'],
                            collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    test_loader = DataLoader(dataset=test_set, batch_size=exp_config['batch_size'],
                                collate_fn=collate_molgraphs, num_workers=args['num_workers'])

    model = load_model(exp_config).to(args['device'])
    loss_criterion = nn.SmoothL1Loss(reduction='none')
    optimizer = Adam(model.parameters(), lr=exp_config['lr'],
                        weight_decay=exp_config['weight_decay'])

    # Path(args['result_path']).mkdir(parents=True, exist_ok=True)

    stopper = EarlyStopping(patience=exp_config['patience'],
                            filename=args['result_path'] + f'/{target}.pth',
                            metric=args['metric'])
    for epoch in range(args['num_epochs']):
        run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)

        # Validation and early stop
        val_score = run_an_eval_epoch(args, model, val_loader)
        early_stop = stopper.step(val_score, model)  # Saves checkpoint to checkpoints/{target}.pth
        print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['metric'],
            val_score, args['metric'], stopper.best_score))

        if early_stop:
            break
