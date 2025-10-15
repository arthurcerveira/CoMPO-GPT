import pickle
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, MolFromInchi
from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from pathlib import Path

import logging

def get_fingerprints(smiles_or_inchi_list, is_inchi=False):
    """Convert SMILES or InChI to Morgan fingerprints."""
    fps = []
    for mol_str in tqdm(smiles_or_inchi_list, desc="Converting to fingerprints"):
        try:
            if is_inchi:
                mol = MolFromInchi(mol_str)
            else:
                mol = Chem.MolFromSmiles(mol_str)
            
            if mol is None:
                continue
                
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
            fps.append(fp)
        except Exception as e:
            logging.warning(f"Failed to process molecule {mol_str}: {str(e)}")
            continue
    
    return np.array(fps)

def train_bbb_model(dataset_path):
    """Train BBB classifier model.
    
    Args:
        dataset_path: Path to BBB.csv file
    Returns:
        Trained RandomForestClassifier or None if training fails
    """
    if not dataset_path.exists():
        logging.error(f"BBB dataset not found at {dataset_path}")
        return None
        
    logging.debug('Processing BBB dataset')
    dataset = pd.read_csv(dataset_path)
    
    # BBB dataset processing
    dataset = dataset.dropna(subset=['activity', 'InChI'])
    dataset = dataset.drop_duplicates(subset=['InChI'])
    
    if len(dataset) < 10:
        logging.info('Less than 10 compounds. Not fitting BBB model')
        return None
    
    # Convert activity to binary and get fingerprints
    y = (dataset['activity'] == 'active').astype(int)
    X = get_fingerprints(dataset['InChI'], is_inchi=True)
    
    if len(X) == 0:
        logging.error('Failed to generate fingerprints for BBB model')
        return None
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    clf.fit(X, y)
    logging.debug(f"BBB Classification score: {clf.score(X, y)}")
    
    return clf

def train_ligand_binding_model(dataset):
    """Train pXC50 regression model."""
    logging.debug(f'Number of obs: {dataset.shape[0]}:')
    logging.debug(f'{dataset.head()}')

    dataset["pXC50"] = pd.to_numeric(dataset["pXC50"])
    dataset = dataset.dropna(subset=['pXC50', 'SMILES'])
    dataset = dataset.drop_duplicates(subset=['SMILES'])

    # Compound-target activities were categorized as active or nonactive
    # where less than 1 μM IC50 value was defined to be active
    dataset['Active'] = dataset['pXC50'] < 6

    print("Total number of compounds:", dataset.shape[0])
    print("Number of active compounds:", dataset['Active'].sum())

    if dataset.shape[0] < 10:
        logging.info('Less than 10 compound-target pairs. Not fitting a model')
        return None
    
    # Get fingerprints and values
    X = get_fingerprints(dataset['SMILES'])
    y = dataset['pXC50'].values

    if len(X) == 0:
        logging.error('Failed to generate fingerprints')
        return None

    # Filter out lines where y is infinite
    valid_idx = np.isfinite(y)
    X = X[valid_idx]
    y = y[valid_idx]

    # Train regressor
    regr = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
    regr.fit(X, y)
    logging.debug(f"pXC50 Regression score: {regr.score(X, y)}")

    return regr


if __name__ == '__main__':
    current_dir = Path(__file__).resolve().parent
    polygon_dir = current_dir.parent.parent
    data_root = Path(polygon_dir / ".." / ".." / 'data')
    
    # Setup output directory
    output_dir = Path(current_dir / "ligand_binding_models")
    output_dir.mkdir(exist_ok=True)

    # Train pXC50 models
    datasets_path = data_root / 'Assays-pXC50'
    datasets = list(datasets_path.glob('*.csv'))
    logging.debug(f'Number of datasets: {len(datasets)}:')

    for path in datasets:
        logging.debug(f'Processing dataset: {path}:')
        dataset = pd.read_csv(path)
        model = train_ligand_binding_model(dataset)
        
        if model is not None:
            with open(output_dir / f'{path.stem}.pkl', 'wb') as f:
                pickle.dump(model, f)
    
    # Train BBB model
    bbb_path = data_root / 'BBB.csv'
    bbb_model = train_bbb_model(bbb_path)
    
    if bbb_model is not None:
        with open(output_dir / 'BBB.pkl', 'wb') as f:
            pickle.dump(bbb_model, f)