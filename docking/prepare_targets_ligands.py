import argparse
import csv
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


def read_target_csv(
    csv_path: Path, active_threshold: float
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Read a target CSV and split rows into active/inactive buckets."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Target file not found: {csv_path}")

    active: List[Tuple[str, float]] = []
    inactive: List[Tuple[str, float]] = []
    seen_smiles = set()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"File {csv_path} has no header row")

        required_columns = {"SMILES", "pXC50"}
        missing_columns = required_columns.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"Missing columns {missing_columns} in {csv_path}")

        for row in reader:
            smiles = (row.get("SMILES") or "").strip()
            if not smiles or smiles.upper() == "SMILES":
                continue

            try:
                pxc50 = float(row.get("pXC50", ""))
            except ValueError:
                continue

            if smiles in seen_smiles:
                continue
            seen_smiles.add(smiles)

            entry = (smiles, pxc50)
            if pxc50 > active_threshold:
                active.append(entry)
            else:
                inactive.append(entry)

    return active, inactive


def sort_molecules_by_pxc50(
    molecules: Sequence[Tuple[str, float]],
    *,
    descending: bool,
) -> List[Tuple[str, float]]:
    """Return molecules sorted by pXC50 in the desired order."""
    return sorted(molecules, key=lambda entry: entry[1], reverse=descending)


def smiles_to_pdbqt(smiles: str, out_path: Path) -> None:
    """Convert a SMILES string into a PDBQT file using RDKit + OpenBabel."""
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


def convert_smiles_to_pdbqt(
    molecules: Sequence[Tuple[str, float]], output_dir: Path, sample_size: int
) -> List[Tuple[str, float]]:
    """Convert SMILES to sequentially named PDBQT files in the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for existing in output_dir.glob("*.pdbqt"):
        existing.unlink()

    prepared: List[Tuple[str, float]] = []
    for smiles, pxc50 in molecules:
        if sample_size > 0 and len(prepared) >= sample_size:
            break

        ligand_index = len(prepared) + 1
        ligand_path = output_dir / f"{ligand_index}.pdbqt"
        try:
            smiles_to_pdbqt(smiles, ligand_path)
        except Exception as exc:
            if ligand_path.exists():
                ligand_path.unlink()
            print(f"  ✗ Failed to convert SMILES '{smiles[:30]}...': {exc}")
            continue

        prepared.append((smiles, pxc50))
        print(f"  ✓ Generated {ligand_path}")

    if not molecules:
        print(f"  No molecules provided for {output_dir}")
    elif sample_size > 0 and len(prepared) < sample_size:
        print(
            f"  ⚠ Requested {sample_size} molecules but only prepared {len(prepared)}."
        )

    return prepared


def write_subset(output_path: Path, rows: Iterable[Tuple[str, float]]) -> None:
    """Persist SMILES / pXC50 rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SMILES", "pXC50"])
        for smiles, pxc50 in rows:
            writer.writerow([smiles, f"{pxc50:.4f}"])


def process_target(
    csv_path: Path,
    output_root: Path,
    sample_size: int,
    active_threshold: float,
) -> Tuple[int, int]:
    active, inactive = read_target_csv(csv_path, active_threshold)

    sorted_active = sort_molecules_by_pxc50(active, descending=True)
    sorted_inactive = sort_molecules_by_pxc50(inactive, descending=False)

    target_dir = output_root / csv_path.stem

    print(f"Converting active molecules for {csv_path.stem}...")
    prepared_active = convert_smiles_to_pdbqt(
        sorted_active, target_dir / "active", sample_size
    )
    print(f"Converting inactive molecules for {csv_path.stem}...")
    prepared_inactive = convert_smiles_to_pdbqt(
        sorted_inactive, target_dir / "inactive", sample_size
    )

    write_subset(target_dir / "active.csv", prepared_active)
    write_subset(target_dir / "inactive.csv", prepared_inactive)

    return len(prepared_active), len(prepared_inactive)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample active/inactive molecules per target from assay CSV files."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data/Assays-pXC50",
        help="Directory containing per-target CSV assay files.",
    )
    parser.add_argument(
        "--ligands-dir",
        type=str,
        default="ligands/targets",
        help="Directory where target subsets will be stored.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of molecules to sample for each of the active/inactive classes.",
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=6.0,
        help="pXC50 threshold above which a molecule is considered active.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    output_root = Path(args.ligands_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"Sampling up to {args.sample_size} actives/inactives from {len(csv_files)} targets...")
    summary = []

    for csv_path in csv_files:
        target_name = csv_path.stem
        if target_name != "D3R":
            print(f"Skipping {target_name}...")
            continue
        print(f"Processing {target_name}...")
        try:
            active_count, inactive_count = process_target(
                csv_path=csv_path,
                output_root=output_root,
                sample_size=args.sample_size,
                active_threshold=args.active_threshold,
            )
        except Exception as exc:
            print(f"✗ {target_name}: {exc}")
            continue

        summary.append((target_name, active_count, inactive_count))
        print(
            f"✓ {target_name}: active={active_count} inactive={inactive_count}"
        )

    if not summary:
        print("No targets were processed successfully.")
        return

    print("\nStored sampled molecules in:")
    print(output_root.resolve())
    print("\nSummary:")
    for target_name, active_count, inactive_count in summary:
        print(f"  {target_name}: active={active_count} inactive={inactive_count}")


if __name__ == "__main__":
    main()
