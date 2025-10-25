from typing import Iterator
import numpy as np
from operator import itemgetter


class MolRLReplayBuffer:
    def __init__(self, max_buffer_size=512, shuffle=False):
        self.max_buffer_size = max_buffer_size
        self.buffer = []
        self.start_index = 0
        self.shuffle = shuffle
        self.indices = np.array([], dtype=np.int64)

    def update_batch(self, states_list, actions, action_log_probs, rewards):
        """
        Store a batch of trajectories in the replay buffer.

        Args:
            states_list: Iterable with one entry per objective. Each entry must
                be an iterable of sequences with the same batch length as
                `actions`.
            actions: Iterable with the sequences that were generated or provided
                as demonstrations.
            action_log_probs: Iterable with the log probabilities associated
                with each action.
            rewards: Iterable with the scalar reward for each action.
        """
        if not states_list:
            return
        # Transpose `states_list` so each element contains one sequence per
        # objective. This keeps sampler logic compact while allowing an
        # arbitrary number of objectives (fixed to five for MPO scenarios).
        transposed_states = list(zip(*states_list))
        new_entries = [
            (tuple(states_per_objective), action, log_prob, reward)
            for states_per_objective, action, log_prob, reward in zip(
                transposed_states, actions, action_log_probs, rewards
            )
        ]
        self.buffer.extend(new_entries)
        if len(self.buffer) > self.max_buffer_size:
            self.buffer = self.buffer[-self.max_buffer_size:]
        self._refresh_indices()

    def __getitem__(self, index):
        return self.buffer[index]

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []
        self.start_index = 0
        self._refresh_indices()

    def restart(self):
        self.start_index = 0
        self._shuffle_indices()

    def iterate_sample(self, mini_batch_size) -> Iterator:
        """
        A mini batch iterator
        """
        for i in range(self.start_index, len(self.buffer), mini_batch_size):
            sampled_indices = self.indices[i:i + mini_batch_size]
            if sampled_indices.size == 0:
                break
            # get sampled batch
            self.start_index = i + mini_batch_size
            yield itemgetter(*sampled_indices)(self.buffer)

    def _refresh_indices(self):
        buffer_len = len(self.buffer)
        self.indices = np.arange(buffer_len, dtype=np.int64)
        self._shuffle_indices()
        self.start_index = 0

    def _shuffle_indices(self):
        if self.shuffle and self.indices.size > 0:
            np.random.shuffle(self.indices)
