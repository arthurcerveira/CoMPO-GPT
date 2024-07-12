#!bin/bash

# Make sure we are in parent directory of script directory
cd "$(dirname "$0")"/..
pwd

# list of target names
targets=("Unconditional" "AChE" "D2R" "D3R" "_5HT2A" "MAOB" "BBB")

# number of epochs comes from command line argument
epochs=$1

# check if epochs is provided
if [ -z "$epochs" ]
then
    echo "Please provide number of epochs as argument"
    exit 1
fi

# create directory to store generated molecules
mkdir -p generated_molecules/${epochs}-epoch

# for each target, generate molecules based on index
for i in {0..6}
do
    # If target is Unconditional, use models_cMolGPT/base.h5
    # if [ $i -eq 0 ]
    # then
    #     model_path="models_cMolGPT/base.h5"
    # else
    #     model_path="models_cMolGPT/finetune.h5_${epochs}"
    # fi

    model_path="models_cMolGPT/finetune.h5_${epochs}"

    # print target name
    echo "Generating molecules for target: ${targets[i]}"
    #print command
    echo "python3 cMolGPT/main.py --mode infer --target ${i} --path ${model_path} \
                          --num_molecules 30000 --output_path generated_molecules/${epochs}-epoch/${targets[i]}.csv"

    python3 cMolGPT/main.py --mode infer --target ${i} --path ${model_path} \
                    --num_molecules 30000 --output_path generated_molecules/${epochs}-epoch/${targets[i]}.csv
done
