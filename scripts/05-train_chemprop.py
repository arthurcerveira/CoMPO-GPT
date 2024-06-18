# conda activate chemprop
import subprocess
# cd to the root of the project
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_DIR + "/..")

targets = ["AChE", "D2R", "D3R", "_5HT2A", "MAOB",] # "BBB",]

# BD: brain disorders
command_template = """
    chemprop_train --data_path ./data/Assays-Inhibition/{target}.tsv \
                   --dataset_type classification \
                   --save_dir ./models_chemprop/{target}-checkpoint \
                   --smiles_column smiles \
                   --target_columns activity
"""

for target in targets:
    command = command_template.format(target=target)
    print(command)
    subprocess.run(command, shell=True)
