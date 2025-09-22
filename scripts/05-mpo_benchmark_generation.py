import subprocess
from itertools import product
# cd to the root of the project
import os
import sys
from concurrent.futures import ProcessPoolExecutor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_DIR + "/..")

EPOCH = 100 if len(sys.argv) < 2 else sys.argv[1]
model_path = f"weights/finetune_mpo.h5_{EPOCH}"
command_template = """
    python CoMPO-GPT/main.py \
      --mode infer \
      --infer_target {conditions_string} \
      --multivariate {agg} --path {model_path} \
      --num_molecules 10000 \
      --output_path generated_molecules/{epochs}-epoch/{disease}_{agg_file}.csv
"""

target_combinations = {
    "Alzheimers": ("AChE", "MAOB", "BBB", "SAScore", "CNSMPO"),
    "Schizophrenia": ("D2R", "_5HT2A", "BBB", "SAScore", "CNSMPO"),
    "Parkinsons": ("D2R", "D3R", "BBB", "SAScore", "CNSMPO"),
}

agg_functions = ("sum", "mean", "max")

def run_command(disease, agg):
    conditions = target_combinations[disease]
    conditions_string = " ".join(conditions)
    command = command_template.format(
        conditions_string=conditions_string, 
        agg=agg, agg_file=agg.upper(),
        model_path=model_path, 
        epochs=EPOCH, disease=disease
    )
    print(f"Executing: {command}")
    subprocess.run(command, shell=True)

# Use ProcessPoolExecutor with a limited number of workers
with ProcessPoolExecutor(max_workers=2) as executor:
    futures = [
        executor.submit(run_command, disease, agg)
        for disease, agg in product(target_combinations.keys(), agg_functions)
    ]

# Wait for all tasks to complete
for future in futures:
    future.result()

print("Done")
