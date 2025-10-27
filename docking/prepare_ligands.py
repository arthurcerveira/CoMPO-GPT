import argparse
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')
sys.path.append(os.path.join(Chem.RDConfig.RDContribDir, 'SA_Score'))
import sascorer  # noqa: E402


CONDITIONS: Dict[str, List[str]] = {
    "alzheimer": ["AChE", "MAOB"],
    "schizophrenia": ["D2R", "_5HT2A"],
    "parkinson": ["D2R", "D3R"],
}

LIGANDS_ROOT = Path("ligands")


def load_smiles_from_file(file_path: Path, random_seed: int = 42) -> List[str]:
    """Load SMILES from a file, filter invalid/unsynthesizable entries and deduplicate."""
    random.seed(random_seed)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    smiles_list = file_path.read_text().splitlines()
    smiles_list = [smi for smi in smiles_list if smi and smi.upper() != "SMILES"]
    if not smiles_list:
        raise ValueError(f"No valid SMILES found in {file_path}")

    print(
        f"Found {len(smiles_list)} molecules in {file_path}, "
        "filtering invalid, unsynthesizable and duplicate molecules..."
    )

    valid_smiles = [smi for smi in smiles_list if Chem.MolFromSmiles(smi) is not None]
    synthesizable_smiles = [
        smi for smi in valid_smiles if sascorer.calculateScore(Chem.MolFromSmiles(smi)) < 6
    ]
    unique_smiles = list(set(synthesizable_smiles))

    random.shuffle(unique_smiles)
    return unique_smiles


def smiles_to_pdbqt(smiles: str, out_path: Path) -> None:
    """Convert a SMILES string to a PDBQT file using RDKit for 3D embedding and obabel for conversion."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)

    tmp_pdb = out_path.with_suffix(".pdb")

    try:
        Chem.MolToPDBFile(mol, str(tmp_pdb))
        cmd = ["obabel", str(tmp_pdb), "-O", str(out_path), "-xh"]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        if tmp_pdb.exists():
            tmp_pdb.unlink()


def convert_smiles_to_pdbqt(smiles_list: List[str], max_ligands: int, output_dir: Path) -> List[Path]:
    """Convert SMILES to sequentially named PDBQT files in the given output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove existing ligand files so numbering stays consistent.
    for existing in output_dir.glob("*.pdbqt"):
        existing.unlink()

    prepared_files: List[Path] = []
    print(f"Converting up to {max_ligands} SMILES to PDBQT files...")

    for idx, smiles in enumerate(smiles_list, start=1):
        if len(prepared_files) >= max_ligands:
            break

        ligand_index = len(prepared_files) + 1
        ligand_pdbqt = output_dir / f"{ligand_index}.pdbqt"

        try:
            smiles_to_pdbqt(smiles, ligand_pdbqt)
        except Exception as exc:
            if ligand_pdbqt.exists():
                ligand_pdbqt.unlink()
            print(f"  ✗ {idx}: Failed to convert {smiles[:30]}... - {exc}")
            continue

        prepared_files.append(ligand_pdbqt)
        print(f"  ✓ {ligand_index}: {smiles[:30]}...")

    print(f"Successfully prepared {len(prepared_files)}/{max_ligands} ligands")
    return prepared_files


def prepare_ligands_for_condition(
    baseline_name: str,
    condition: str,
    smiles_file: Path,
    output_root: Path,
    sample_size: int,
    random_seed: int,
) -> List[Path]:
    """Prepare ligands for a single condition and return the generated file paths."""
    try:
        smiles_list = load_smiles_from_file(smiles_file, random_seed)
    except Exception as exc:
        print(f"Skipping {smiles_file}: {exc}")
        return []

    output_dir = output_root / f"{baseline_name}_{condition}"
    prepared_files = convert_smiles_to_pdbqt(smiles_list, sample_size, output_dir)

    if not prepared_files:
        print(f"No ligands were prepared for {baseline_name} / {condition}")
    else:
        print(f"Prepared ligands stored in {output_dir}")

    return prepared_files


def process_baseline(
    baseline_name: str,
    generated_molecules_dir: Path,
    output_root: Path,
    sample_size: int,
    random_seed: int,
) -> Dict[str, List[Path]]:
    baseline_dir = generated_molecules_dir / baseline_name
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")

    print(f"\n=== Preparing ligands for baseline: {baseline_name} ===")
    summary: Dict[str, List[Path]] = {}

    for condition, targets in CONDITIONS.items():
        targets_str = "_".join(targets)
        smiles_file = baseline_dir / f"{condition}.csv"
        if baseline_dir.name == "25-epoch":
            smiles_file = baseline_dir / f"{targets_str}_SUM.csv"

        print(f"\n--- Condition: {condition} ---")
        print(f"Source file: {smiles_file}")

        prepared_files = prepare_ligands_for_condition(
            baseline_name=baseline_name,
            condition=condition,
            smiles_file=smiles_file,
            output_root=output_root,
            sample_size=sample_size,
            random_seed=random_seed,
        )
        summary[condition] = prepared_files

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ligands for docking by converting SMILES to PDBQT.")
    parser.add_argument("--baseline", type=str, help="Specific baseline to process (optional)")
    parser.add_argument("--sample-size", type=int, default=200, help="Number of ligands to prepare per condition")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for SMILES shuffling")
    parser.add_argument(
        "--generated-molecules-dir",
        type=str,
        default="../generated_molecules",
        help="Directory containing generated molecules",
    )
    parser.add_argument(
        "--ligands-dir",
        type=str,
        default=str(LIGANDS_ROOT),
        help="Output directory for prepared ligands",
    )

    args = parser.parse_args()

    generated_molecules_dir = Path(args.generated_molecules_dir)
    ligands_dir = Path(args.ligands_dir)

    if not generated_molecules_dir.exists():
        print(f"Error: Generated molecules directory not found: {generated_molecules_dir}")
        sys.exit(1)

    if args.baseline:
        baselines = [args.baseline]
    else:
        baselines = [d.name for d in generated_molecules_dir.iterdir() if d.is_dir()]

    if not baselines:
        print("No baselines found to process.")
        sys.exit(0)

    all_results: Dict[str, Dict[str, List[Path]]] = {}
    for baseline_name in baselines:
        try:
            summary = process_baseline(
                baseline_name=baseline_name,
                generated_molecules_dir=generated_molecules_dir,
                output_root=ligands_dir,
                sample_size=args.sample_size,
                random_seed=args.random_seed,
            )
            all_results[baseline_name] = summary
        except Exception as exc:
            print(f"Error preparing ligands for {baseline_name}: {exc}")
            continue

    print("\n" + "=" * 50)
    print("PREPARATION SUMMARY")
    print("=" * 50)

    for baseline_name, baseline_results in all_results.items():
        print(f"\nBaseline: {baseline_name}")
        for condition, prepared_files in baseline_results.items():
            print(f"  {condition}: {len(prepared_files)} ligands prepared")
            if prepared_files:
                sample_file = prepared_files[0]
                print(f"    Example ligand: {sample_file}")

    print(f"\nPrepared ligands root directory: {ligands_dir.resolve()}")


if __name__ == "__main__":
    main()
