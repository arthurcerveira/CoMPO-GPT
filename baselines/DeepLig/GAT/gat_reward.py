import torch
import numpy as np
from pathlib import Path

from dgllife.model import GATPredictor
from moleculenet.utils import predict

from train_gat import args, node_featurizer, smiles_to_g


current_dir = Path(__file__).resolve().parent


def load_gat(target, args):
    model = GATPredictor(in_feats=node_featurizer.feat_size(), n_tasks=1).to(args['device'])
    state_dict = torch.load(current_dir / 'checkpoints' / f'{target}.pth')['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_gat_on_graphs(graphs, model, args):
    predictions = list()

    for mol_graph in graphs:
        if mol_graph is None:
            predictions.append(np.nan)
            continue

        prediction = predict(args, model, mol_graph)
        predictions.append(prediction.detach().cpu().numpy()[0][0])

    return predictions


class GATReward:
    def __init__(self, targets):
        self.targets = targets
        self.models = [load_gat(target, args) for target in targets]

    def __call__(self, smiles, predictor=None, invalid_reward=0.0):
        if not isinstance(smiles, str):
            print("Not individual SMILES", type(smiles))

        mol, prop, nan_smiles = self.predict([smiles])

        if len(nan_smiles) == 1:
            return invalid_reward

        return np.exp(prop[0] / 3)

    def predict(self, trajectory):
        """
        Mean molecular activity prediction across all targets
        """
        single_trajectory = isinstance(trajectory, str)
        if single_trajectory:
            trajectory = [trajectory]
        graphs = [smiles_to_g(smiles) for smiles in trajectory]
        is_nan = np.array([graph is None for graph in graphs])

        predictions = np.array(
            [run_gat_on_graphs(graphs, model, args) for model in self.models]
        )
        try:
            nan_smiles = np.array(trajectory)[is_nan]
            not_nan_smiles = np.array(trajectory)[~is_nan]
        except Exception as e:
            print(e)
            breakpoint()
            # nan_smiles = np.array(trajectory)[is_nan]
            # not_nan_smiles = np.array(trajectory)[~is_nan]
        prediction = np.mean(predictions, axis=0)
        prediction[is_nan] = 0.0

        # Clip the prediction to be between 0 and 10
        prediction = np.clip(prediction, 0, 10)

        # If running on a single trajectory, return the prediction
        if single_trajectory:
            return not_nan_smiles, prediction[0], 

        return not_nan_smiles, prediction, nan_smiles


if __name__ == '__main__':
    smiles = [
        'C=C1C(=O)OCC12CCC(O)C(COC)OC2c1ccccc1',
        'C=CNc1cccc2c(CCCS(=O)(=O)CCCC(=O)O)c[nH]c12',
        'CC(=O)Cn1cnc(C(=O)C2=CN(CC(N)=O)S(=O)(=O)c3ccc(Cl)cc32)c1Cl',
        'CC(=O)NC1=NC(=O)C(C(C#N)C2CC2)=CN1',
        'CC(C)CC(COCc1c(O)ccc2cc(O)ccc12)C(=O)O',
        'CC(C)CCc1c(OCC(=O)O)cn(CC(C)C)c1NC(=O)c1cn(C)cn1',
        'CC(CNCC=NN1C(=O)c2cccc(c3cc(Cl)ccc3CN2)C1=O)c1ccc(Cl)cc1',
        'CC1(C)COC2C(C(NNC(=O)N3CCCC3c3ccccc3)c3ccccc3)CC(O)C2O1',
        'CC1=NN(C(=O)c2ccc(Cl)cc2)C(c2ccc(-c3ccccc3)cc2)N1',
        'CC1CCC(=O)C2NC(N)=NC12',
    ]

    get_reward = GATReward(['D2R', 'D3R'])
    not_nan_smiles, prediction, nan_smiles = get_reward.predict(smiles)
    for smiles, pred in zip(smiles, prediction):
        print(f"{smiles:60s} | {pred}")
