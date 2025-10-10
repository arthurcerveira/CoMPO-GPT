import subprocess
from pathlib import Path
from itertools import product

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem


# Directories
receptors_dir = Path("receptors")
ligands_dir = Path("ligands")
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)


# List of SMILES
smiles_list = [
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Example: Ibuprofen
    "C1=CC=C(C=C1)C2=CC=CC=C2"         # Example: Biphenyl
]


# Convert SMILES to 3D PDBQT
def smiles_to_pdbqt(smiles, out_path):
    mol = Chem.MolFromSmiles(smiles)
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


# Dock each ligand against all receptors
receptors = list(receptors_dir.glob("*.pdbqt"))
smiles_receptors = list(product(smiles_list, receptors))
pbar = tqdm(smiles_receptors, total=len(smiles_receptors))

for smi, receptor_file in pbar:
    smi_idx = smiles_list.index(smi)
    ligand_name = f"lig_{smi_idx+1}"
    ligand_pdbqt = ligands_dir / f"{ligand_name}.pdbqt"
    ligands_dir.mkdir(exist_ok=True)
    smiles_to_pdbqt(smi, ligand_pdbqt)

    receptor_name = receptor_file.stem
    output_file = results_dir / f"{ligand_name}_to_{receptor_name}.pdbqt"
    config_file = f"receptors/{receptor_name}_vina_config.txt"

    pbar.set_description(f"Docking {ligand_name} → {receptor_name}")
    # Run Vina
    cmd = [
        "vina",
        "--receptor", str(receptor_file),
        "--ligand", str(ligand_pdbqt),
        "--config", config_file,
        "--out", str(output_file),
        # "--cpu", 16,
        # "--log", str(output_file.with_suffix(".log"))
    ]
    # Save stdout to log file
    with open(output_file.with_suffix(".log"), "w") as log_file:
        subprocess.run(cmd, check=True, stdout=log_file, stderr=log_file)

    # print(f"Docking finished: {ligand_name} → {receptor_name}")
