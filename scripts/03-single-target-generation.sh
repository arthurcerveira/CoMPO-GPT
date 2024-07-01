#!bin/bash

# Make sure we are in parent directory of script directory
cd "$(dirname "$0")"/..
pwd

# list of target names
targets=("Unconditional" "AChE" "D2R" "D3R" "_5HT2A" "MAOB" "BBB")

model_path="models_cMolGPT/finetune.h5_20"

# for each target, generate molecules based on index
for i in {0..6}
do
    # print target name
    echo "Generating molecules for target: ${targets[i]}"
    #print command
    echo "python3 cMolGPT/main.py --mode infer --target ${i} --path ${model_path} \
                          --num_molecules 30000 --output_path generated_molecules/${targets[i]}.csv"

    python3 cMolGPT/main.py --mode infer --target ${i} --path ${model_path} \
                    --num_molecules 30000 --output_path generated_molecules/${targets[i]}.csv
done
