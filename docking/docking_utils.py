import subprocess
from pathlib import Path
from itertools import product


# Directories
receptors_dir = Path("receptors")
ligands_dir = Path("ligands")
results_dir = Path("poses")
results_dir.mkdir(exist_ok=True)


def generate_docking_commands(pdbqt_files, target_receptors, output_dir, ligand_prefix="lig"):
    """
    Generate docking commands for pre-converted PDBQT files.
    
    Args:
        pdbqt_files: List of PDBQT file paths
        target_receptors: List of receptor names (without .pdbqt extension)
        output_dir: Directory to save results
        ligand_prefix: Prefix for ligand names (used as baseline name)
    
    Returns:
        List of command dictionaries with all necessary information
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    pdbqt_receptors = list(product(pdbqt_files, receptor_files))
    
    commands = []
    for ligand_pdbqt, receptor_file in pdbqt_receptors:
        # Extract ligand index from filename (e.g., "1.pdbqt" -> 1)
        ligand_idx = int(ligand_pdbqt.stem)
        ligand_name = f"{ligand_prefix}_{ligand_idx}"
        
        receptor_name = receptor_file.stem
        output_file = output_dir / f"{ligand_name}_to_{receptor_name}.pdbqt"
        config_file = f"receptors/{receptor_name}_vina_config.txt"
        log_file = output_file.with_suffix(".log")
        
        # Create the command dictionary
        cmd_dict = {
            'ligand_pdbqt': ligand_pdbqt,
            'ligand_name': ligand_name,
            'receptor_file': receptor_file,
            'receptor_name': receptor_name,
            'output_file': output_file,
            'config_file': config_file,
            'log_file': log_file
        }
        commands.append(cmd_dict)
    
    return commands


def generate_bash_script(commands, output_dir, max_workers=1, script_name="run_docking.sh"):
    """
    Generate a bash script for parallel docking execution.
    
    Args:
        commands: List of command dictionaries from generate_docking_commands
        output_dir: Directory to save the script and results
        max_workers: Maximum number of parallel workers
        script_name: Name of the generated bash script
    
    Returns:
        Path to the generated bash script
    """
    output_dir = Path(output_dir)
    script_path = output_dir / script_name
    
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated docking script\n")
        f.write(f"# Total tasks: {len(commands)}\n")
        f.write(f"# Max workers: {max_workers}\n")
        f.write(f"# Generated on: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}\n\n")
        
        f.write("set -e  # Exit on any error\n\n")

        # f.write('VINA_EXEC="/home/aacerveira/docking/vina_1.2.7_linux_x86_64"\n')
        f.write('VINA_EXEC="vina"\n')
        f.write("# Function to run a single docking task\n")
        f.write("run_docking_task() {\n")
        f.write("    local ligand_pdbqt=\"$1\"\n")
        f.write("    local receptor_file=\"$2\"\n")
        f.write("    local output_file=\"$3\"\n")
        f.write("    local config_file=\"$4\"\n")
        f.write("    local log_file=\"$5\"\n")
        f.write("    local ligand_name=\"$6\"\n")
        f.write("    local receptor_name=\"$7\"\n\n")
        
        f.write("    echo \"Starting docking: $ligand_name → $receptor_name\"\n")
        f.write("    \n")
        f.write("    # Run Vina (ligand PDBQT file already exists)\n")
        f.write("    $VINA_EXEC --receptor \"$receptor_file\" --ligand \"$ligand_pdbqt\" --config \"$config_file\" --out \"$output_file\" --verbosity 2 > \"$log_file\" 2>&1\n")
        f.write("    \n")
        f.write("    echo \"Completed: $ligand_name → $receptor_name\"\n")
        f.write("}\n\n")
        
        f.write("# Array to store background job PIDs\n")
        f.write("declare -a pids=()\n\n")
        
        f.write("echo \"Starting docking with $max_workers workers...\"\n")
        f.write("start_time=$(date +%s)\n\n")
        
        # Generate the parallel execution logic
        for i, cmd in enumerate(commands):
            f.write(f"# Task {i+1}: {cmd['ligand_name']} → {cmd['receptor_name']}\n")
            f.write(f"run_docking_task \"{cmd['ligand_pdbqt']}\" \"{cmd['receptor_file']}\" \"{cmd['output_file']}\" \"{cmd['config_file']}\" \"{cmd['log_file']}\" \"{cmd['ligand_name']}\" \"{cmd['receptor_name']}\" &\n")
            f.write(f"pids+=($!)\n")
            
            # Add wait logic every max_workers tasks
            if (i + 1) % max_workers == 0:
                f.write(f"\n# Wait for batch of {max_workers} tasks to complete\n")
                f.write("for pid in \"${pids[@]}\"; do\n")
                f.write("    wait $pid\n")
                f.write("done\n")
                f.write("pids=()  # Clear the array\n")
                f.write(f"echo \"Completed batch {i//max_workers + 1}\"\n\n")
        
        # Wait for remaining tasks
        if len(commands) % max_workers != 0:
            f.write("# Wait for remaining tasks to complete\n")
            f.write("for pid in \"${pids[@]}\"; do\n")
            f.write("    wait $pid\n")
            f.write("done\n")
            f.write("echo \"Completed final batch\"\n\n")
        
        f.write("end_time=$(date +%s)\n")
        f.write("duration=$((end_time - start_time))\n")
        f.write("echo \"All docking tasks completed in ${duration} seconds\"\n")
        f.write("echo \"Results saved to: $(pwd)/$(basename \"$0\" .sh)_results/\"\n")
    
    # Make the script executable
    script_path.chmod(0o755)
    
    return script_path


def docking_experiment_script(pdbqt_files, target_receptors, output_dir, ligand_prefix="lig", max_workers=None):
    """
    Create docking experiment scripts for prepared PDBQT ligands against specified receptors.
    
    Args:
        pdbqt_files: List of PDBQT file paths
        target_receptors: List of receptor names (without .pdbqt extension)
        output_dir: Directory to save results
        ligand_prefix: Prefix for ligand names
        max_workers: Maximum number of parallel workers (None for auto-detect)
    
    Returns:
        Script path
    """
    output_dir = Path(output_dir)
    
    # Determine number of workers
    if max_workers is None:
        max_workers = 1
    
    if not pdbqt_files:
        raise ValueError("No prepared ligand files were provided")

    print(f"Processing {len(pdbqt_files)} molecules against {len(target_receptors)} targets")
    print(f"Total docking tasks: {len(pdbqt_files) * len(target_receptors)}")
    print(f"Max workers: {max_workers}")
    
    # Generate docking commands
    commands = generate_docking_commands(pdbqt_files, target_receptors, output_dir, ligand_prefix)
    
    # Generate bash script
    script_name = f"docking_{ligand_prefix}_{len(commands)}_tasks.sh"
    script_path = generate_bash_script(commands, output_dir, max_workers, script_name)
    
    print(f"Generated bash script: {script_path}")
    print("Script generation complete. Run the script manually to execute docking.")
    return script_path


# Example usage (commented out)
if __name__ == "__main__":
    # List of SMILES
    smiles_list = [
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",   # Example: Ibuprofen
        "C1=CC=C(C=C1)C2=CC=CC=C2"         # Example: Biphenyl
    ]
    
    # Run docking for all receptors
    receptors = [f.stem for f in receptors_dir.glob("*.pdbqt")]
    docking_experiment_script(smiles_list, 2, receptors, results_dir)
