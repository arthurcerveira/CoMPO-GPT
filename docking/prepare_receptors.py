# apt install openbabel python3-openbabel
# pip install biopython
# conda install -c conda-forge openbabel vina
import os
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select
import requests
import subprocess


OUTPUT_DIR = Path("receptors")
OUTPUT_DIR.mkdir(exist_ok=True)

# PDB IDs for each target
TARGETS = {
    "_5HT2A": "6A93",          # 5-HT2A receptor (with risperidone)
    "D2R": "6CM4",           # Dopamine D2 receptor
    "D3R": "3PBL",           # Dopamine D3 receptor
    "AChE": "4EY7",           # Human acetylcholinesterase
    "MAOB": "2V5Z",           # Monoamine oxidase B
}

# Known cofactors to preserve if present
KNOWN_COFACS = {
    "FAD", "FMN", "NAD", "NADH", "NADP",
    "HEM", "HEME", "HBA", "COA",
    "ZN", "FE", "MG", "MN", "CA", "CU"
}


class ProteinCofactorSelect(Select):
    """Keep only protein residues and essential cofactors."""
    def __init__(self):
        self.preserved = set()

    def accept_residue(self, residue):
        resname = residue.get_resname().strip()
        # Protein residues
        if residue.id[0] == " ":
            return True
        # Keep essential cofactors
        if resname in KNOWN_COFACS:
            self.preserved.add(resname)
            return True
        return False


def download_pdb(pdb_id, out_path):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"Downloading {pdb_id} from RCSB...")
    r = requests.get(url)
    r.raise_for_status()
    with open(out_path, "w") as f:
        f.write(r.text)


def clean_pdb(infile, outfile):
    # Remove ligands, ions, waters, keeping only protein atoms.
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", infile)

    selector = ProteinCofactorSelect()
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(outfile), selector)

    return selector.preserved


def convert_to_pdbqt(pdb_path, pdbqt_path):
    cmd = ["obabel", pdb_path, "-O", pdbqt_path, "-xr", "-xh"]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


for name, pdb_id in TARGETS.items():
    base = OUTPUT_DIR / name
    pdb_file = Path(str(base) + ".pdb")
    pdb_clean = Path(str(base) + "_clean.pdb")
    pdbqt_file = Path(str(base) + ".pdbqt")

    if not pdb_file.exists():
        download_pdb(pdb_id, pdb_file)
    else:
        print(f"{pdb_id} already downloaded.")

    preserved = clean_pdb(pdb_file, pdb_clean)
    if preserved:
        print(f"Preserved cofactors: {', '.join(sorted(preserved))}")

    convert_to_pdbqt(pdb_clean, pdbqt_file)

    # Sanity check
    with open(pdbqt_file) as f:
        content = f.read()
        if "ROOT" in content or "BRANCH" in content:
            print(f"Warning: {name} receptor contains ROOT/BRANCH (ligand not removed).")
        else:
            print(f"{name} receptor ready: {pdbqt_file}")

print("\nAll receptors processed and ready for docking.")
