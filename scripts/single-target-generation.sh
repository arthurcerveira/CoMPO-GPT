#!bin/bash

# Make sure we are in parent directory of script directory
cd "$(dirname "$0")"/..
pwd

# list of target names
targets=("Unconditional") # "AChE" "BBB" "D2R" "D3R" "_5HT2A" "MAOB")

# for each target, generate molecules based on index
for i in {0..0} #..6}
do
    # target index is i
    target_index=${i}

    # print target name
    echo "Generating molecules for target: ${targets[i]}"
    #print command
    echo "python3 main.py --mode infer --target ${target_index} --path models_cMolGPT/finetune.h5 \
                          --num_molecules 30000 --output_path generated_molecules/${targets[i]}.csv"

    python3 main.py --mode infer --target ${target_index} --path models_cMolGPT/finetune.h5 \
                    --num_molecules 30000 --output_path generated_molecules/${targets[i]}.csv
done
