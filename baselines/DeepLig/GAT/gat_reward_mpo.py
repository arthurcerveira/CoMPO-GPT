from pathlib import Path

import numpy as np
from dgllife.model import GATPredictor
from moleculenet.utils import predict

from train_gat import regression_args, classification_args, node_featurizer, smiles_to_g
from gat_reward import load_gat, run_gat_on_graphs

import sys
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir / ".." / ".." / ".." / "benchmarks"))
from guacamol.common_scoring_functions import (
    CNS_MPO_ScoringFunction, 
    SyntheticAccessibilityScoringFunction
)

import os
from rdkit import Chem
sys.path.append(os.path.join(Chem.RDConfig.RDContribDir, 'SA_Score'))
import sascorer


class GATRewardMPO:
    def __init__(self, targets):
        self.targets = targets
        self.target_models = [load_gat(target, regression_args) for target in targets]
        self.bbb_model = load_gat("BBB", classification_args)
        self.cns_mpo_scoring_function = CNS_MPO_ScoringFunction()
        self.sascorer_function = SyntheticAccessibilityScoringFunction(
            sascorer.calculateScore
        )

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

        target_predictions = np.array([
            run_gat_on_graphs(graphs, model, regression_args) 
            for model in self.target_models
        ])
        # Divide target predictions by 10 to get the probability of the active class
        target_predictions = target_predictions / 10
        bbb_prediction = np.array(
            [run_gat_on_graphs(graphs, self.bbb_model, classification_args)]
        )
        cns_mpo_prediction = []
        sascorer_prediction = []
        for smiles in trajectory:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                cns_mpo_prediction.append(np.nan)
                sascorer_prediction.append(np.nan)
                continue
            cns_mpo_prediction.append(self.cns_mpo_scoring_function.score_mol(mol))
            sascorer_prediction.append(self.sascorer_function.score_mol(mol))

        cns_mpo_prediction = np.array([cns_mpo_prediction])
        sascorer_prediction = np.array([sascorer_prediction])

        try:
            nan_smiles = np.array(trajectory)[is_nan]
            not_nan_smiles = np.array(trajectory)[~is_nan]
        except Exception as e:
            print(e)
            breakpoint()
            # nan_smiles = np.array(trajectory)[is_nan]
            # not_nan_smiles = np.array(trajectory)[~is_nan]

        all_scores = np.concatenate([
            target_predictions, bbb_prediction, cns_mpo_prediction, sascorer_prediction
        ], axis=0)
        prediction = np.mean(all_scores, axis=0)
        prediction[is_nan] = 0.0

        # No need to clip the prediction as the scores are already normalized
        # prediction = np.clip(prediction, 0, 10)

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

    get_reward = GATRewardMPO(['D2R', 'D3R'])
    not_nan_smiles, prediction, nan_smiles = get_reward.predict(smiles)
    for smiles, pred in zip(smiles, prediction):
        print(f"{smiles:60s} | {pred}")
