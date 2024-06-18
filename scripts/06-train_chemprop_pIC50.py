# conda activate chemprop
import subprocess
# cd to the root of the project
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_DIR + "/..")

targets = ["AChE", "D2R", "D3R", "_5HT2A", "MAOB"]

# BD: brain disorders
command_template = """
    chemprop_train --data_path ./data/Assays-IC50/{target}.tsv \
                   --dataset_type regression \
                   --save_dir ./models_chemprop/{target}-IC50-checkpoint \
                   --smiles_column smiles \
                   --target_columns IC50
"""

for target in targets:
    command = command_template.format(target=target)
    print(command)
    subprocess.run(command, shell=True)
