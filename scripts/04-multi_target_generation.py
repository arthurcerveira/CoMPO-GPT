import subprocess
import itertools
# cd to the root of the project
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_DIR + "/..")

targets_to_idx = {
    "AChE": 1,
    "D2R": 2,
    "D3R": 3,
    "_5HT2A": 4,
    "MAOB": 5,
}

model_path = "models_cMolGPT/finetune.h5_20"

command_template = """
    python3 cMolGPT/main.py \
      --mode infer --infer_target {t1} {t2} \
      --multivariate {agg} --path {model_path} \
      --num_molecules 30000 \
      --output_path generated_molecules/{combination}_{agg_file}.csv
"""

target_combinations = (
    # Alzheimers
    ("AChE", "MAOB"),
    # Schizophrenia
    ("D2R", "_5HT2A"),
    # Parkinsons
    ("D2R", "D3R"),
)

# agg_functions = ("mean", "max") #, "sum")
agg_functions = ("sum", "mean", "max")

for combination, agg in itertools.product(target_combinations, agg_functions):
    t1 = targets_to_idx[combination[0]]
    t2 = targets_to_idx[combination[1]]

    command = command_template.format(
        t1=t1, t2=t2, agg=agg, combination="_".join(combination),
        agg_file=agg.upper(), model_path=model_path
    )
    print(command)
    subprocess.run(command, shell=True)

print("Done")
