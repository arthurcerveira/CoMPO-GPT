import random
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import sys
import os

from rdkit import RDLogger, Chem
RDLogger.DisableLog('rdApp.*')
sys.path.append(os.path.join(Chem.RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from docking_utils import docking_experiment_script


SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
CONDITIONS = {
    "alzheimer": ["AChE", "MAOB"],
    "schizophrenia": ["D2R", "_5HT2A"],
    "parkinson": ["D2R", "D3R"],
}


def load_smiles_from_file(file_path: Path, random_seed: int = 42) -> List[str]:
    random.seed(random_seed)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    smiles_list = file_path.read_text().splitlines()
    smiles_list = [smi for smi in smiles_list if smi and smi.upper() != 'SMILES']
    if len(smiles_list) == 0:
        raise ValueError(f"No valid SMILES found in {file_path}")

    print(
        f"Found {len(smiles_list)} molecules in {file_path}, " + 
        "filtering invalid, unsynthesizable and duplicate molecules..."
    )

    # Filter invalid and unsynthesizable molecules
    valid_smiles = [smi for smi in smiles_list if Chem.MolFromSmiles(smi) is not None]
    synthesizable_smiles = [
        smi for smi in valid_smiles if sascorer.calculateScore(Chem.MolFromSmiles(smi)) < 6
    ]
    unique_smiles = list(set(synthesizable_smiles))
    
    # Randomize the order of the molecules
    random.shuffle(unique_smiles)
    return unique_smiles


def run_docking_for_baseline(baseline_name: str, generated_molecules_dir: Path, 
                           output_base_dir: Path, sample_size: int = 100, 
                           random_seed: int = 42, max_workers: int = None,) -> Dict[str, List[Path]]:
    """
    Run docking experiments for all conditions in a baseline.
    
    Args:
        baseline_name: Name of the baseline
        generated_molecules_dir: Directory containing generated molecules
        output_base_dir: Base directory for output
        sample_size: Number of molecules to sample per condition
        random_seed: Random seed for reproducible sampling
        max_workers: Maximum number of parallel workers (None for auto-detect)
    
    Returns:
        Dictionary mapping condition names to lists of output files or script paths
    """
    baseline_dir = generated_molecules_dir / baseline_name
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")
    
    print(f"\n=== Processing baseline: {baseline_name} ===")
    
    results = {}
    
    for condition, targets in CONDITIONS.items():
        print(f"\n--- Processing condition: {condition} ---")
        print(f"Targets: {targets}")
        
        # Find condition files
        # condition_files = find_condition_files(baseline_dir, condition)
        file_path = baseline_dir / f"{condition}.csv"
        if baseline_dir.name == "25-epoch":
            targets_string = "_".join(targets)
            file_path = baseline_dir / f"{targets_string}_SUM.csv"
        
        print(f"Processing file: {file_path}")
        
        try:
            # Load and sample SMILES
            smiles_list = load_smiles_from_file(file_path, random_seed)
            
            # Create output directory
            file_stem = file_path.stem
            output_dir = output_base_dir / baseline_name / f"{condition}"
            ligand_prefix = f"{baseline_name}_{condition}"
            
            # Run docking experiment
            print(f"Running docking for {len(smiles_list)} molecules against {len(targets)} targets")
            output_files = docking_experiment_script(
                smiles_list=smiles_list,
                total_smiles=sample_size,
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
            print(f"Error processing {file_path}: {e}")
            continue
    
    return results


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description="Run docking pipeline for generated molecules")
    parser.add_argument("--baseline", type=str, help="Specific baseline to process (optional)")
    parser.add_argument("--sample-size", type=int, default=200, help="Number of molecules to sample per condition")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--output-dir", type=str, default="./poses", help="Output directory for results")
    parser.add_argument("--generated-molecules-dir", type=str, default="../generated_molecules", 
                       help="Directory containing generated molecules")
    parser.add_argument("--max-workers", type=int, default=8, 
                       help="Maximum number of parallel workers")

    args = parser.parse_args()
    
    # Set up paths
    generated_molecules_dir = Path(args.generated_molecules_dir)
    output_base_dir = Path(args.output_dir)
    
    if not generated_molecules_dir.exists():
        print(f"Error: Generated molecules directory not found: {generated_molecules_dir}")
        sys.exit(1)
    
    # Get list of baselines to process
    if args.baseline:
        baselines = [args.baseline]
    else:
        baselines = [d.name for d in generated_molecules_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(baselines)} baseline(s) to process: {baselines}")
    
    # Process each baseline
    all_results = {}
    for baseline_name in baselines:
        try:
            results = run_docking_for_baseline(
                baseline_name=baseline_name,
                generated_molecules_dir=generated_molecules_dir,
                output_base_dir=output_base_dir,
                sample_size=args.sample_size,
                random_seed=args.random_seed,
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