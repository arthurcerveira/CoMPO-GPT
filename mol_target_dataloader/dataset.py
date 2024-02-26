# coding=utf-8

"""
Implementation of a SMILES dataset.
"""

import torch
import torch.utils.data as tud


class Dataset(tud.Dataset):
    """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""

    def __init__(self, smiles_list,target_list, vocabulary, tokenizer):
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer
        self._smiles_list = list(smiles_list)
        self._target_list = list(target_list)


    def __getitem__(self, i):
        smi = self._smiles_list[i]
        target = self._target_list[i]
        tokens = self._tokenizer.tokenize(smi)
        encoded = self._vocabulary.encode(tokens)
        return [torch.tensor(encoded, dtype=torch.long), target] # pylint: disable=E1102

    def __len__(self):
        return len(self._smiles_list)

    @staticmethod
    def collate_fn(encoded_pairs):
        """Converts a list of encoded sequences into a padded tensor"""
        encoded_seqs, targets = list(zip(*encoded_pairs))
        targets = [int(tmp) for tmp in targets]
        max_length = max([seq.size(0) for seq in encoded_seqs])
        collated_arr = torch.zeros(len(encoded_seqs), max_length, dtype=torch.long)  # padded with zeroes
        for i, seq in enumerate(encoded_seqs):
            collated_arr[i, :seq.size(0)] = seq
        return (collated_arr, targets)

