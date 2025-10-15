import sys
sys.path.append('./release/')

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm, trange
import pickle
from pathlib import Path
from rdkit import Chem, DataStructs

from stackRNN import StackAugmentedRNN
from data import GeneratorData
from utils import canonical_smiles


use_cuda = torch.cuda.is_available()
gen_data_path = './data/chembl_22_clean_1576904_sorted_std_final.smi'
tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']

gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t', 
                         cols_to_read=[0], keep_header=True, tokens=tokens)

hidden_size = 1500  # As defined in the paper
stack_width = 512   # As defined in the paper
# stack_width = 1500   # Default
stack_depth = 200   # Default
layer_type = 'GRU'  # As defined in the paper
lr = 0.001          # Default
optimizer_instance = torch.optim.Adadelta

my_generator = StackAugmentedRNN(input_size=gen_data.n_characters, hidden_size=hidden_size,
                                 output_size=gen_data.n_characters, layer_type=layer_type,
                                 n_layers=1, is_bidirectional=False, has_stack=True,
                                 stack_width=stack_width, stack_depth=stack_depth, 
                                 use_cuda=use_cuda, 
                                 optimizer_instance=optimizer_instance, lr=lr)

model_path = './checkpoints/generator/checkpoint_biggest_rnn_stack_512'


if __name__ == '__main__':
    # epochs = 1_000  # As defined in the paper
    dataset_size = 1_000_000  # Roughly 1,000,000 molecules, as defined in the paper
    epochs = 1_000  # As defined in the paper
    iterations = dataset_size * epochs  # fit method runs one iter
    losses = my_generator.fit(
        gen_data, iterations, plot_every=1000, print_every=5000, patience=50
    )
    my_generator.evaluate(gen_data)
    my_generator.save_model(model_path)
    print("Model saved to: ", model_path)
