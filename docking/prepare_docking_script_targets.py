import argparse
import sys
from pathlib import Path
from typing import Dict, List

from docking_utils import docking_experiment_script

DEFAULT_GROUPS = ("active", "inactive")


def collect_prepared_ligands(ligand_dir: Path) -> List[Path]:
    """Return prepared ligand files sorted by numeric index"""
    if not ligand_dir.exists():
        raise FileNotFoundError(f"Ligand directory not found: {ligand_dir}")

    indexed_files = []
    for path in ligand_dir.glob("*.pdbqt"):
        try:
            index = int(path.stem)
        except ValueError:
            continue
        indexed_files.append((index, path))

    if not indexed_files:
        raise ValueError(f"No prepared ligand files found in {ligand_dir}")

    indexed_files.sort(key=lambda item: item[0])

    return [path for _, path in indexed_files]


def run_docking_for_target(
    target_name: str,
    ligands_root: Path,
    output_base_dir: Path,
    max_workers: int,
    groups: List[str],
) -> Dict[str, Path]:
    """Generate docking scripts for the provided target across all requested ligand groups."""
    print(f"\n=== Processing target: {target_name} ===")

    target_dir = ligands_root / target_name
    if not target_dir.exists():
        raise FileNotFoundError(f"Target ligands directory not found: {target_dir}")

    scripts: Dict[str, Path] = {}
    for group in groups:
        print(f"\n--- Group: {group} ---")
        ligand_dir = target_dir / group
        try:
            pdbqt_files = collect_prepared_ligands(ligand_dir)
        except Exception as exc:
            print(f"Skipping group '{group}' for {target_name}: {exc}")
            continue

        print(f"Found {len(pdbqt_files)} ligands for {target_name} ({group})")

        output_dir = output_base_dir / target_name / group
        ligand_prefix = f"{target_name}_{group}"

        script_path = docking_experiment_script(
            pdbqt_files=pdbqt_files,
            target_receptors=[target_name],
            output_dir=output_dir,
            ligand_prefix=ligand_prefix,
            max_workers=max_workers,
        )
        scripts[group] = script_path
        print(f"Docking script generated: {script_path}")

    if not scripts:
        raise RuntimeError(f"No docking scripts were generated for target {target_name}")

    return scripts


def discover_targets(ligands_root: Path) -> List[str]:
    """Return a sorted list of target directories under ligands_root."""
    targets = sorted(
        entry.name
        for entry in ligands_root.iterdir()
        if entry.is_dir()
    )
    if not targets:
        raise FileNotFoundError(f"No target directories found in {ligands_root}")
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate docking scripts for prepared target ligands."
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Specific target to process (default: all targets found in ligands directory).",
    )
    parser.add_argument(
        "--ligands-dir",
        type=str,
        default="ligands/targets",
        help="Directory containing per-target ligand folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="poses/targets",
        help="Directory where docking scripts/results will be written.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of parallel docking workers.",
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="*",
        default=list(DEFAULT_GROUPS),
        help="Ligand groups to process (default: active inactive).",
    )

    args = parser.parse_args()

    ligands_root = Path(args.ligands_dir)
    if not ligands_root.exists():
        print(f"Error: ligands directory not found: {ligands_root}")
        sys.exit(1)

    output_base_dir = Path(args.output_dir)
    groups = args.groups or list(DEFAULT_GROUPS)

    if args.target:
        targets = [args.target]
    else:
        try:
            targets = discover_targets(ligands_root)
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

    print(f"Found {len(targets)} target(s) to process: {targets}")

    all_results: Dict[str, Dict[str, Path]] = {}
    for target_name in targets:
        try:
            scripts = run_docking_for_target(
                target_name=target_name,
                ligands_root=ligands_root,
                output_base_dir=output_base_dir,
                max_workers=args.max_workers,
                groups=groups,
            )
            all_results[target_name] = scripts
        except Exception as exc:
            print(f"Error processing target {target_name}: {exc}")
            continue

    print("\n" + "=" * 50)
    print("DOCKING SCRIPT GENERATION SUMMARY")
    print("=" * 50)
    for target_name, scripts in all_results.items():
        print(f"\nTarget: {target_name}")
        for group_name, script_path in scripts.items():
            print(f"  {group_name}: {script_path}")

    if all_results:
        print(f"\nScripts written under: {output_base_dir.resolve()}")
        print("Run the generated .sh files to execute docking tasks.")
    else:
        print("\nNo scripts were generated. Check logs above for details.")


if __name__ == "__main__":
    main()
