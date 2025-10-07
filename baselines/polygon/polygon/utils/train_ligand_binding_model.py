import pickle
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from pathlib import Path

import logging

def train_ligand_binding_model(dataset):
    logging.debug(f'Number of obs: {dataset.shape[0]}:')
    logging.debug(f'{dataset.head()}')

    dataset["pXC50"] = pd.to_numeric(dataset["pXC50"])
    dataset = dataset.dropna(subset=['pXC50','SMILES'])
    dataset = dataset.drop_duplicates(subset=['SMILES'])

    # Compound-target activities were categorized as active or nonactive, where less than 1 μM IC50 value was defined to be active
    # Select active compounds smiles strings
    dataset['Active'] = dataset['pXC50'] < 6

    print("Total number of compounds:", dataset.shape[0])
    print("Number of active compounds:", dataset['Active'].sum())

    active_smiles = dataset.loc[dataset['Active'], 'SMILES']

    if dataset.shape[0]<10:
        logging.info('Less than 10 compound-target pairs. Not fitting a model')
        return 1
    
    # convert to fingerprint
    fps = []
    values = []
    for x,y in tqdm(dataset[['SMILES','pXC50']].values):
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x),2)
        except:
            continue
        
        fps.append(fp)
        values.append(y)

    X = np.array(fps)
    y = np.array(values)

    # Filter out lines where y is infinite
    valid_idx = np.isfinite(y)
    X = X[valid_idx]
    y = y[valid_idx]

    regr = RandomForestRegressor(n_estimators=1000,random_state=0,n_jobs=-1)
    regr.fit(X,y)
    regr.score(X,y)

    logging.debug(regr.score(X,y))

    return regr, active_smiles


if __name__ == '__main__':
    current_dir = Path(__file__).resolve().parent
    polygon_dir = current_dir.parent.parent
    datasets_path = Path(polygon_dir / ".." / ".." / 'data' / 'Assays-pXC50/')
    output_dir = Path(current_dir / "ligand_binding_models")
    output_dir.mkdir(exist_ok=True)

    datasets = list(datasets_path.glob('*.csv'))
    logging.debug(f'Number of datasets: {len(datasets)}:')

    for path in datasets:
        logging.debug(f'Processing dataset: {path}:')
        dataset = pd.read_csv(path)
        model, active_smiles = train_ligand_binding_model(dataset)

        with open(output_dir / f'{path.stem}.pkl', 'wb') as f:
            pickle.dump(model, f)
        # Save active smiles to txt
        active_smiles.to_csv(output_dir / f'{path.stem}_active_smiles.txt', index=False, header=False)
