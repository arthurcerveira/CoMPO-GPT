import subprocess
import itertools
# cd to the root of the project
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_DIR + "/..")

targets_to_idx = {
    "AChE": 1,
    "BBB": 2,
    "D2R": 3,
    "D3R": 4,
    "_5HT2A": 5,
    "MAOB": 6,
}

# bd: brain disorders

command_template = """
    python3 main.py \
      --mode infer --infer_target {t1} {t2} \
      --multivariate {agg} --path models_cMolGPT/finetune.h5 \
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

agg_functions = ("mean", "max") #, "sum")

for combination, agg in itertools.product(target_combinations, agg_functions):
    t1 = targets_to_idx[combination[0]]
    t2 = targets_to_idx[combination[1]]

    command = command_template.format(
        t1=t1, t2=t2, agg=agg, combination="_".join(combination),
        agg_file=agg.upper()
    )
    print(command)
    subprocess.run(command, shell=True)

print("Done")
