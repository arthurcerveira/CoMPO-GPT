import subprocess
from pathlib import Path
from itertools import product

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem


# Directories
receptors_dir = Path("receptors")
ligands_dir = Path("ligands")
results_dir = Path("poses")
results_dir.mkdir(exist_ok=True)


def smiles_to_pdbqt(smiles, out_path):
    """Convert SMILES to 3D PDBQT format using RDKit and Open Babel."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    tmp_pdb = out_path.with_suffix(".pdb")
    Chem.MolToPDBFile(mol, tmp_pdb)
    
    # Convert PDB -> PDBQT using Open Babel
    cmd = ["obabel", str(tmp_pdb), "-O", str(out_path), "-xh"]
    # Suppress stdout
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    tmp_pdb.unlink()  # remove temporary PDB


def run_docking_experiment(smiles_list, target_receptors, output_dir, ligand_prefix="lig"):
    """
    Run docking experiments for a list of SMILES against specified receptors.
    
    Args:
        smiles_list: List of SMILES strings
        target_receptors: List of receptor names (without .pdbqt extension)
        output_dir: Directory to save results
        ligand_prefix: Prefix for ligand names
    
    Returns:
        List of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ligands_dir = output_dir / "ligands"
    ligands_dir.mkdir(exist_ok=True)
    
    # Get receptor files
    receptor_files = []
    for receptor_name in target_receptors:
        receptor_file = receptors_dir / f"{receptor_name}.pdbqt"
        if receptor_file.exists():
            receptor_files.append(receptor_file)
        else:
            print(f"Warning: Receptor file not found: {receptor_file}")
    
    if not receptor_files:
        raise ValueError("No valid receptor files found")
    
    # Create all combinations
    smiles_receptors = list(product(smiles_list, receptor_files))
    pbar = tqdm(smiles_receptors, total=len(smiles_receptors))
    
    output_files = []
    
    for smi, receptor_file in pbar:
        smi_idx = smiles_list.index(smi)
        ligand_name = f"{ligand_prefix}_{smi_idx+1}"
        ligand_pdbqt = ligands_dir / f"{ligand_name}.pdbqt"
        
        try:
            smiles_to_pdbqt(smi, ligand_pdbqt)
        except Exception as e:
            print(f"Error converting SMILES {smi}: {e}")
            continue

        receptor_name = receptor_file.stem
        output_file = output_dir / f"{ligand_name}_to_{receptor_name}.pdbqt"
        config_file = f"receptors/{receptor_name}_vina_config.txt"

        pbar.set_description(f"Docking {ligand_name} → {receptor_name}")
        
        # Run Vina
        cmd = [
            "vina",
            "--receptor", str(receptor_file),
            "--ligand", str(ligand_pdbqt),
            "--config", config_file,
            "--out", str(output_file),
            "--verbosity", "2",
        ]
        
        # Save stdout to log file
        log_file = output_file.with_suffix(".log")
        try:
            with open(log_file, "w") as f:
                subprocess.run(cmd, check=True, stdout=f, stderr=f)
            output_files.append(output_file)
        except subprocess.CalledProcessError as e:
            print(f"Error running Vina for {ligand_name} → {receptor_name}: {e}")
            continue
    
    return output_files


# Example usage (commented out)
if __name__ == "__main__":
    # List of SMILES
    smiles_list = [
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Example: Ibuprofen
        "C1=CC=C(C=C1)C2=CC=CC=C2"         # Example: Biphenyl
    ]
    
    # Run docking for all receptors
    receptors = [f.stem for f in receptors_dir.glob("*.pdbqt")]
    run_docking_experiment(smiles_list, receptors, results_dir)
