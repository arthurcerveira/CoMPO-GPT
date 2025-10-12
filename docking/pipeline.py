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

from run_docking import run_docking_experiment


CONDITIONS = {
    "alzheimer": ["AChE", "MAOB"],
    "schizophrenia": ["D2R", "_5HT2A"],
    "parkinson": ["D2R", "D3R"],
}


def load_smiles_from_file(file_path: Path, sample_size: int = 100, random_seed: int = 42) -> List[str]:
    random.seed(random_seed)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    smiles_list = file_path.read_text().splitlines()
    smiles_list = [smi for smi in smiles_list if smi and smi.upper() != 'SMILES']
    if len(smiles_list) == 0:
        raise ValueError(f"No valid SMILES found in {file_path}")

    print(
        f"Found {len(smiles_list)} molecules in {file_path}, " + 
        "filtering invalid and unsynthesizable molecules..."
    )

    # Filter invalid and unsynthesizable molecules
    valid_smiles = [smi for smi in smiles_list if Chem.MolFromSmiles(smi) is not None]
    synthesizable_smiles = [
        smi for smi in valid_smiles if sascorer.calculateScore(Chem.MolFromSmiles(smi)) < 6
    ]
    
    # Sample molecules
    if len(synthesizable_smiles) <= sample_size:
        print(f"Warning: Only {len(synthesizable_smiles)} molecules available, using all")
        return synthesizable_smiles
    else:
        sampled = random.sample(synthesizable_smiles, sample_size)
        print(f"Sampled {len(sampled)} molecules from {len(synthesizable_smiles)} available")
        return sampled


def find_condition_files(baseline_dir: Path, condition: str) -> List[Path]:
    """
    Find all files for a given condition in a baseline directory.
    
    Args:
        baseline_dir: Path to baseline directory
        condition: Condition name (alzheimer, schizophrenia, parkinson)
    
    Returns:
        List of file paths for the condition
    """
    condition_files = []
    
    # Look for files with condition name
    for pattern in [f"{condition}.csv", f"{condition.title()}.csv", f"{condition.upper()}.csv"]:
        file_path = baseline_dir / pattern
        if file_path.exists():
            condition_files.append(file_path)
    
    # Look for files with condition-specific patterns
    if condition == "alzheimer":
        patterns = ["AChE_MAOB_SUM.csv", "Alzheimers_SUM.csv"]
    elif condition == "schizophrenia":
        patterns = ["D2R__5HT2A_SUM.csv", "Schizophrenia_SUM.csv"]
    elif condition == "parkinson":
        patterns = ["D2R_D3R_SUM.csv", "Parkinsons_SUM.csv"]
    else:
        patterns = []
    
    for pattern in patterns:
        condition_files.extend(baseline_dir.glob(pattern))
    
    return list(set(condition_files))  # Remove duplicates


def run_docking_for_baseline(baseline_name: str, generated_molecules_dir: Path, 
                           output_base_dir: Path, sample_size: int = 100, 
                           random_seed: int = 42) -> Dict[str, List[Path]]:
    """
    Run docking experiments for all conditions in a baseline.
    
    Args:
        baseline_name: Name of the baseline
        generated_molecules_dir: Directory containing generated molecules
        output_base_dir: Base directory for output
        sample_size: Number of molecules to sample per condition
        random_seed: Random seed for reproducible sampling
    
    Returns:
        Dictionary mapping condition names to lists of output files
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
        condition_files = find_condition_files(baseline_dir, condition)
        
        if not condition_files:
            print(f"Warning: No files found for condition {condition} in {baseline_dir}")
            continue
        
        print(f"Found {len(condition_files)} file(s) for condition {condition}")
        
        # Process each file for this condition
        for file_path in condition_files:
            print(f"Processing file: {file_path.name}")
            
            try:
                # Load and sample SMILES
                smiles_list = load_smiles_from_file(file_path, sample_size, random_seed)
                
                # Create output directory
                file_stem = file_path.stem
                output_dir = output_base_dir / baseline_name / f"{condition}_{file_stem}"
                
                # Run docking experiment
                print(f"Running docking for {len(smiles_list)} molecules against {len(targets)} targets")
                output_files = run_docking_experiment(
                    smiles_list=smiles_list,
                    target_receptors=targets,
                    output_dir=output_dir,
                    ligand_prefix=f"{baseline_name}_{condition}"
                )
                
                if condition not in results:
                    results[condition] = []
                results[condition].extend(output_files)
                
                print(f"Completed docking: {len(output_files)} successful dockings")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    return results


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description="Run docking pipeline for generated molecules")
    parser.add_argument("--baseline", type=str, help="Specific baseline to process (optional)")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of molecules to sample per condition")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--output-dir", type=str, default="./poses", help="Output directory for results")
    parser.add_argument("--generated-molecules-dir", type=str, default="../generated_molecules", 
                       help="Directory containing generated molecules")
    
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
                random_seed=args.random_seed
            )
            all_results[baseline_name] = results
        except Exception as e:
            print(f"Error processing baseline {baseline_name}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*50)
    print("PIPELINE SUMMARY")
    print("="*50)
    
    for baseline_name, baseline_results in all_results.items():
        print(f"\nBaseline: {baseline_name}")
        for condition, output_files in baseline_results.items():
            print(f"  {condition}: {len(output_files)} successful dockings")
    
    print(f"\nResults saved to: {output_base_dir}")


if __name__ == "__main__":
    main()