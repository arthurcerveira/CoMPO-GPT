import torch
from torch.nn.utils.rnn import pad_sequence


class RewardFunction:
    def __init__(self, vocab, rewardnet, device, num_objectives=5):
        self.rewardnet = rewardnet
        self.vocab = vocab
        self.device = device
        self.num_objectives = num_objectives

    def fit(self, demo_smiles_list, model_smiles):
        if not model_smiles:
            return 0.0
        self.rewardnet.train()
        loss = self.get_reward_gru(demo_smiles_list, model_smiles)['loss']
        self.rewardnet.optimizer.zero_grad()
        loss.backward()
        self.rewardnet.optimizer.step()
        return loss.item()

    def _encode_smiles_batch(self, smiles_batch):
        tensor_batch = [
            torch.tensor(self.vocab.encode(string, add_bos=True, add_eos=True), dtype=torch.long, device=self.device)
            for string in smiles_batch
        ]
        if not tensor_batch:
            return torch.empty((0, 1), dtype=torch.long, device=self.device)
        return pad_sequence(tensor_batch, batch_first=True, padding_value=self.vocab.pad)

    def get_reward_gru(self, demo_smiles_list, samples):
        if len(demo_smiles_list) != self.num_objectives:
            raise ValueError(f"Expected {self.num_objectives} objective sets, got {len(demo_smiles_list)}")

        demo_pad_ids_list = [self._encode_smiles_batch(smiles) for smiles in demo_smiles_list]
        samples_pad_ids = self._encode_smiles_batch(samples)
        if samples_pad_ids.size(0) == 0:
            zero_reward = torch.zeros(0, device=self.device)
            return {"reward": zero_reward, "loss": torch.tensor(0.0, device=self.device)}

        results = self.rewardnet.compute_reward_gru(demo_pad_ids_list, samples_pad_ids)
        return results

    def get_reward(self, demo_smiles_list, samples):
        return self.get_reward_gru(demo_smiles_list, samples)['reward']
