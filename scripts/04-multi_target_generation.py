import subprocess
import itertools
# cd to the root of the project
import os
import sys
from concurrent.futures import ProcessPoolExecutor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_DIR + "/..")

EPOCH = 100 if len(sys.argv) < 2 else sys.argv[1]
model_path = f"weights/finetune_mpo.h5_{EPOCH}"
command_template = """
    python3 CoMPO-GPT/main.py \
      --mode infer --infer_target {t1} {t2} \
      --multivariate {agg} --path {model_path} \
      --num_molecules 10000 \
      --output_path generated_molecules/{epochs}-epoch/{combination}_{agg_file}.csv
"""

target_combinations = (
    # Alzheimers
    ("AChE", "MAOB"),
    # Schizophrenia
    ("D2R", "_5HT2A"),
    # Parkinsons
    ("D2R", "D3R"),
)

agg_functions = ("sum", "mean", "max")

def run_command(combination, agg):
    t1, t2 = combination
    command = command_template.format(
        t1=t1, t2=t2, agg=agg, combination="_".join(combination),
        agg_file=agg.upper(), model_path=model_path, epochs=EPOCH
    )
    print(f"Executing: {command}")
    subprocess.run(command, shell=True)

# Use ProcessPoolExecutor with a limited number of workers
with ProcessPoolExecutor(max_workers=12) as executor:
    futures = [
        executor.submit(run_command, combination, agg)
        for combination, agg in itertools.product(target_combinations, agg_functions)
    ]

# Wait for all tasks to complete
for future in futures:
    future.result()

print("Done")
