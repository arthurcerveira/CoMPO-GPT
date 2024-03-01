#!bin/bash

# python3 main.py --mode infer --infer_target 1 2 --multivariate sum --path model_finetune.h5 \
#                 --num_molecules 30000 --output_path generated_molecules/EGFR_HTR1A_SUM.csv

# python3 main.py --mode infer --infer_target 1 3 --multivariate sum --path model_finetune.h5 \
#                 --num_molecules 30000 --output_path generated_molecules/EGFR_S1PR1_SUM.csv

# python3 main.py --mode infer --infer_target 2 3 --multivariate sum --path model_finetune.h5 \
#                 --num_molecules 30000 --output_path generated_molecules/HTR1A_S1PR1_SUM.csv

python3 main.py --mode infer --infer_target 1 2 --multivariate max --path model_finetune.h5 \
                --num_molecules 30000 --output_path generated_molecules/EGFR_HTR1A_MAX.csv

python3 main.py --mode infer --infer_target 1 3 --multivariate max --path model_finetune.h5 \
                --num_molecules 30000 --output_path generated_molecules/EGFR_S1PR1_MAX.csv

python3 main.py --mode infer --infer_target 2 3 --multivariate max --path model_finetune.h5 \
                --num_molecules 30000 --output_path generated_molecules/HTR1A_S1PR1_MAX.csv