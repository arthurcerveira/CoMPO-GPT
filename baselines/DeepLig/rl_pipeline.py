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
from reinforcement import Reinforcement

sys.path.append('./GAT/')
from GAT import GATReward


use_cuda = torch.cuda.is_available()
gen_data_path = './data/chembl_22_clean_1576904_sorted_std_final.smi'
tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']

gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t', 
                         cols_to_read=[0], keep_header=True, tokens=tokens)

def plot_hist(prediction, n_to_generate):
    print("Mean value of predictions:", prediction.mean())
    print("Proportion of valid SMILES:", len(prediction)/n_to_generate)

hidden_size = 1500  # As defined in the paper
stack_width = 512   # As defined in the paper
# stack_width = 1500  # Default
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
# model_path = './checkpoints/generator/checkpoint_biggest_rnn'
my_generator.load_model(model_path)

disease_to_targets = {
    "schizophrenia": ("D2R", "_5HT2A"),
    "alzheimer": ("AChE", "MAOB"),
    "parkinson": ("D2R", "D3R"),
}

current_dir = Path(__file__).resolve().parent
output_path = current_dir / ".." / ".." / "generated_molecules" / "DeepLig-100-512"
output_path.mkdir(exist_ok=True)


def estimate_and_update(generator, predict_reward, n_to_generate, return_unique=True, **kwargs):
    generated = []
    pbar = tqdm(range(n_to_generate))
    for i in pbar:
        pbar.set_description("Generating molecules...")
        generated.append(generator.evaluate(gen_data, predict_len=120)[1:-1])

    sanitized = canonical_smiles(generated, sanitize=False, throw_warning=False)[:-1]
    unique_smiles = sanitized
    if return_unique:
        # Removing [1:] as it raises an error when all smiles are duplicates
        unique_smiles = list(np.unique(sanitized))  # [1:]

    smiles, prediction, nan_smiles = predict_reward.predict(unique_smiles)  

    plot_hist(prediction, n_to_generate)
        
    return smiles, prediction


def generate_n_molecules(generator, n_to_generate):
    generated = []
    pbar = tqdm(range(n_to_generate))
    pbar.set_description("Generating molecules...")

    for i in pbar:
        generated.append(generator.evaluate(gen_data, predict_len=120)[1:-1])
    
    return generated


def simple_moving_average(previous_values, new_value, ma_window_size=10):
    value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
    value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
    return value_ma


if __name__ == '__main__':
    disease = sys.argv[1]  # "options: schizophrenia, alzheimer, parkinson"
    targets = disease_to_targets[disease]
    predict_reward = GATReward(targets)
    
    n_to_generate = 200   # Default
    n_policy_replay = 10  # Default (apparently unused)
    n_policy = 15         # Default
    # n_iterations = 1500   # As defined in the paper
    n_iterations = 100   # Paper defined as 1500: 100 iterations * 15 policies

    my_generator_max = StackAugmentedRNN(input_size=gen_data.n_characters, 
                                        hidden_size=hidden_size,
                                        output_size=gen_data.n_characters, 
                                        layer_type=layer_type,
                                        n_layers=1, is_bidirectional=False, has_stack=True,
                                        stack_width=stack_width, stack_depth=stack_depth, 
                                        use_cuda=use_cuda, 
                                        optimizer_instance=optimizer_instance, lr=lr)

    my_generator_max.load_model(model_path)
    RL_max = Reinforcement(my_generator_max, predictor=None, get_reward=predict_reward)

    rewards_max = []
    rl_losses_max = []

    for i in range(n_iterations):
        for j in trange(n_policy, desc='Policy gradient...'):
            # cur_reward, cur_loss = RL_max.policy_gradient(gen_data, get_features=get_fp)
            cur_reward, cur_loss = RL_max.policy_gradient(gen_data)
            rewards_max.append(simple_moving_average(rewards_max, cur_reward)) 
            rl_losses_max.append(simple_moving_average(rl_losses_max, cur_loss))

        print(f'Policy {i+1}/{n_iterations} finished')
        print('Average reward: ', rewards_max[-1])
        print('Average loss: ', rl_losses_max[-1])

        smiles_cur, prediction_cur = estimate_and_update(RL_max.generator, 
                                                        predict_reward,  # my_predictor, 
                                                        n_to_generate,)  # get_features=get_fp)
        print('Sample trajectories:')
        for sm in smiles_cur[:5]:
            print(sm)

    smiles_biased_max = generate_n_molecules(RL_max.generator, 10_000)
    with open(output_path / f'{disease}.csv', 'w') as f:
        f.write('\n'.join(smiles_biased_max))
