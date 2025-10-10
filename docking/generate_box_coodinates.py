"""
Prepare receptor structures for AutoDock Vina and automatically determine docking boxes.
"""
from pathlib import Path
from Bio.PDB import PDBParser
import numpy as np
import json
from vina import Vina
import os


CONFIG_TEMPLATE = """
receptor = {receptor_path}
ligand = {ligand_path}

center_x = {center[0]:.3f}
center_y = {center[1]:.3f}
center_z = {center[2]:.3f}

size_x = {size[0]:.3f}
size_y = {size[1]:.3f}
size_z = {size[2]:.3f}

exhaustiveness = {exhaustiveness}
num_modes = {num_modes}
"""

# Known cofactors to preserve
KNOWN_COFAC = {
    "FAD", "NAD", "NDP", "FMN", "HEM", "COA", "ZN", "MG", "MN", "FE", "CA"
}


def get_center_of_atoms(atoms):
    coords = np.array([atom.coord for atom in atoms])
    return coords.mean(axis=0)


def get_box_size(atoms, padding=5.0):
    coords = np.array([atom.coord for atom in atoms])
    min_c, max_c = coords.min(axis=0), coords.max(axis=0)
    size = (max_c - min_c) + padding
    return size


def detect_ligand_and_cofactor(structure):
    cofactors, ligands = list(), list()
    for model in structure:
        for chain in model:
            for residue in chain:
                het = residue.id[0].strip()
                resname = residue.resname.strip()
                if het and len(residue) > 1:
                    if resname in KNOWN_COFAC:
                        cofactors.append(residue)
                    elif resname not in ["HOH"]:  # not water
                        ligands.append(residue)
    return ligands, cofactors


def compute_box_from_residues(residues):
    atoms = [atom for res in residues for atom in res.get_atoms()]
    center = get_center_of_atoms(atoms)
    size = get_box_size(atoms)
    return center, size


def determine_docking_box(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", pdb_file)
    ligands, cofactors = detect_ligand_and_cofactor(structure)

    if ligands:
        print(f"Docking box determined from ligand: {ligands[0].resname}")
        return compute_box_from_residues(ligands)
    elif cofactors:
        print(f"Docking box determined from cofactor: {cofactors[0].resname}")
        return compute_box_from_residues(cofactors)
    else:
        # fallback to geometric center of protein
        protein_atoms = [atom for atom in structure.get_atoms() if atom.parent.id[0] == " "]
        print("Docking box determined from receptor geometric center (no ligand/cofactor found).")
        return compute_box_from_residues([atom.parent for atom in protein_atoms])


def main():
    targets = {
        "_5HT2A": "6A93",
        "D2R": "6CM4",
        "D3R": "3PBL",
        "AChE": "4EY7",
        "MAOB": "2V5Z",
    }

    data_dir = Path("receptors")
    box_info = dict()

    for name, pdb_id in targets.items():
        pdb_path = data_dir / f"{name}_clean.pdb"
        pdbqt_path = data_dir / f"{name}.pdbqt"

        if not pdb_path.exists():
            print(f"Missing cleaned PDB for {name}, skipping.")
            continue

        # Compute docking box
        center, size = determine_docking_box(pdb_path)
        box_info[name] = {
            "center": [float(x) for x in center],
            "size": [float(s) for s in size],
        }

        print(f"{name} docking box: center={center.round(2)}, size={size.round(2)}")

        # Write config file
        config_file = data_dir / f"{name}_vina_config.txt"
        config_file.write_text(CONFIG_TEMPLATE.format(
            receptor_path=pdbqt_path, ligand_path="<ligand_path_here>",  # Placeholder
            center=center, size=size, exhaustiveness=8, num_modes=9
        ).strip() + "\n")
        print(f"Config file written to {config_file}")

    # Save box parameters for downstream docking
    with open(data_dir / "docking_boxes.json", "w") as f:
        json.dump(box_info, f, indent=2)

    print("\nDocking box information saved to docking_boxes.json")


if __name__ == "__main__":
    main()