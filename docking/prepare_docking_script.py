from pathlib import Path
from typing import Dict, List
import argparse
import sys

from docking_utils import docking_experiment_script


# SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
CONDITIONS = {
    "alzheimer": ["AChE", "MAOB"],
    "schizophrenia": ["D2R", "_5HT2A"],
    "parkinson": ["D2R", "D3R"],
}


def collect_prepared_ligands(ligand_dir: Path, sample_size: int) -> List[Path]:
    """Return prepared ligand files sorted by numeric index, limited by sample size."""
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
    if sample_size > 0:
        indexed_files = indexed_files[:sample_size]

    return [path for _, path in indexed_files]


def run_docking_for_baseline(
    baseline_name: str,
    ligands_root: Path,
    output_base_dir: Path,
    sample_size: int = 100,
    max_workers: int = None,
) -> Dict[str, List[Path]]:
    """
    Run docking experiments for all conditions in a baseline using pre-prepared ligands.

    Args:
        baseline_name: Name of the baseline
        ligands_root: Root directory containing prepared ligands
        output_base_dir: Base directory for output
        sample_size: Number of ligands to include per condition
        max_workers: Maximum number of parallel workers (None for auto-detect)

    Returns:
        Dictionary mapping condition names to lists of generated script paths
    """
    print(f"\n=== Processing baseline: {baseline_name} ===")
    
    results = {}
    
    for condition, targets in CONDITIONS.items():
        print(f"\n--- Processing condition: {condition} ---")
        print(f"Targets: {targets}")
        
        ligand_prefix = f"{baseline_name}_{condition}"
        ligand_dir = ligands_root / ligand_prefix
        print(f"Expecting prepared ligands in: {ligand_dir}")
        
        try:
            pdbqt_files = collect_prepared_ligands(ligand_dir, sample_size)
            print(f"Found {len(pdbqt_files)} prepared ligands for docking")
            
            # Create output directory
            output_dir = output_base_dir / baseline_name / f"{condition}"
            
            # Run docking experiment
            print(f"Running docking for {len(pdbqt_files)} molecules against {len(targets)} targets")
            output_files = docking_experiment_script(
                pdbqt_files=pdbqt_files,
                target_receptors=targets,
                output_dir=output_dir,
                ligand_prefix=ligand_prefix,  # Use baseline name for ligand directory structure
                max_workers=max_workers,
            )
            
            if condition not in results:
                results[condition] = []
            results[condition].append(output_files)
            
            print(f"Docking script generated: {output_files}")
        except Exception as e:
            print(f"Error processing prepared ligands for {condition}: {e}")
            continue
    
    return results


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description="Run docking pipeline for generated molecules")
    parser.add_argument("--baseline", type=str, help="Specific baseline to process (optional)")
    parser.add_argument("--sample-size", type=int, default=200, help="Number of molecules to sample per condition")
    parser.add_argument("--output-dir", type=str, default="./poses", help="Output directory for results")
    parser.add_argument("--ligands-dir", type=str, default="./ligands",
                       help="Directory containing prepared ligand folders")
    parser.add_argument("--max-workers", type=int, default=8, 
                       help="Maximum number of parallel workers")

    args = parser.parse_args()
    
    # Set up paths
    output_base_dir = Path(args.output_dir)
    ligands_root = Path(args.ligands_dir)
    
    if not ligands_root.exists():
        print(f"Error: Prepared ligands directory not found: {ligands_root}")
        sys.exit(1)
    
    # Get list of baselines to process
    if args.baseline:
        baselines = [args.baseline]
    else:
        baselines = set()
        for entry in ligands_root.iterdir():
            if not entry.is_dir():
                continue
            for condition in CONDITIONS.keys():
                suffix = f"_{condition}"
                if entry.name.endswith(suffix):
                    baseline = entry.name[: -len(suffix)]
                    if baseline:
                        baselines.add(baseline)
                    break
        baselines = sorted(baselines)
    
    print(f"Found {len(baselines)} baseline(s) to process: {baselines}")
    
    # Process each baseline
    all_results = {}
    for baseline_name in baselines:
        try:
            results = run_docking_for_baseline(
                baseline_name=baseline_name,
                ligands_root=ligands_root,
                output_base_dir=output_base_dir,
                sample_size=args.sample_size,
                max_workers=args.max_workers,
            )
            all_results[baseline_name] = results
        except Exception as e:
            print(f"Error processing baseline {baseline_name}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*50)
    print("PIPELINE SUMMARY")
    print("="*50)
    
    print("Mode: Script Generation Only")
    for baseline_name, baseline_results in all_results.items():
        print(f"\nBaseline: {baseline_name}")
        for condition, script_paths in baseline_results.items():
            print(f"  {condition}: {len(script_paths)} scripts generated")
            for script_path in script_paths:
                print(f"    - {script_path}")
    print(f"\nScripts saved to: {output_base_dir}")
    print("\nTo execute the scripts, run them manually or use:")
    print("  find . -name '*.sh' -exec bash {} \\;")


if __name__ == "__main__":
    main()
