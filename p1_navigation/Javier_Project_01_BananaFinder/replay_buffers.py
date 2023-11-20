import numpy as np
import random

import torch

from collections import deque, namedtuple


class ReplayBuffer():
    """Simple buffer that stores previously experienced steps."""

    def __init__(self, action_size, buffer_size, batch_size):

        self.action_size = action_size
        self.batch_size = batch_size

        # Create a double-ended queue of size "buffer_size"
        self.memory = deque(maxlen=buffer_size)

        # Define a tuple to store passed experiences
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        # Check CUDA is available & set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # Create an experienced tuple and append it to the memory
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)

    def sample(self):
        """Take a sample from memory."""
        exps = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([exp.state for exp in exps if exp is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([exp.action for exp in exps if exp is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in exps if exp is not None])).float().to(self.device)

        next_states = torch.from_numpy(
            np.vstack([exp.next_state for exp in exps if exp is not None])).float().to(self.device)

        dones = torch.from_numpy(np.vstack([exp.done for exp in exps if exp is not None]
                                           ).astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
