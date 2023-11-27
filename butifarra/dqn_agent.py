import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e3)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 1.00            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-5               # learning rate
UPDATE_EVERY = 4        # how often to update the network
EPSILON = 1e-5          # epsilon for priority replay buffer
ALPHA = .5              # alpha for priority replay buffer
BETA = .5               # beta for priority replay buffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RandomAgent():
    def __init__(self, *args, suffix='random', **kwargs):
        self.writer = SummaryWriter(comment=suffix)
        self.t_epsiode = 0

    def act(self, state, eps=0., action_space=None):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        mask = None
        if isinstance(state, dict):
            if 'action_mask' in state:
                mask = state['action_mask']
        return action_space.sample(mask)

    def step(self, *args, **kwargs):
        pass

    def report_score(self, score):
        self.writer.add_scalar('score', score, self.t_epsiode)
        self.t_epsiode += 1


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, suffix='dqn', use_mask=False):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.writer = SummaryWriter(comment=suffix)
        self.use_mask = use_mask

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR, weight_decay=1e-5)

        # Replay memory
        self.memory = PriorityReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.t_epsiode = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        if isinstance(self.memory, PriorityReplayBuffer):
            priority = self.compute_priority(state, action, reward, next_state)
            self.memory.add(state, action, reward, next_state, done, priority)
        else:
            self.memory.add(state, action, reward, next_state, done)

        self.t_step += 1
        if self.t_step % UPDATE_EVERY == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def compute_priority(self, state, action, reward, next_state):
        nest_state_pred = self.qnetwork_target(torch.from_numpy(next_state).to(device))
        error = reward +\
            GAMMA * nest_state_pred.max()
        if action is not None:
            state_pred = self.qnetwork_local(torch.from_numpy(state).to(device))
            error -= state_pred[action]
        priority = abs(error) + EPSILON
        return priority.item()

    def report_score(self, score):
        self.writer.add_scalar('score', score, self.t_epsiode)
        self.t_epsiode += 1

    def act(self, state, eps=0., action_space=None):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        mask = None
        if isinstance(state, dict):
            if 'action_mask' in state:
                mask = state['action_mask']
            state = state['observation']
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            action_values = action_values.cpu().data.numpy().flatten()
            if mask is not None and self.use_mask:
                action_values[~mask.astype(bool)] = -np.inf
            return np.argmax(action_values)
        else:
            return action_space.sample(mask)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        try:
            states, actions, rewards, next_states, dones, probs, indices = experiences
        except ValueError:
            states, actions, rewards, next_states, dones = experiences
        N = states.shape[0]

        target_pred = self.qnetwork_local(next_states)
        best_actions = target_pred.argmax(dim=1)

        Q_targets_next = self.qnetwork_target(next_states)[torch.arange(N), best_actions.flatten()]
        target = rewards.flatten() + gamma * dones.flatten().logical_not() * Q_targets_next

        prediction = self.qnetwork_local(states)[torch.arange(N), actions.flatten()]

        if isinstance(self.memory, PriorityReplayBuffer):
            error = (prediction - target).abs()
            priorities = error + EPSILON
            self.memory.update_priorities(indices, priorities)
            weights = torch.from_numpy((1 / N / probs)**BETA).to(device)
            loss = (weights * (prediction - target.detach())**2).mean()
        else:
            loss = F.mse_loss(prediction, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar('loss', loss, self.t_step)

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PriorityReplayBuffer():
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=[
                "state",
                "action",
                "reward",
                "next_state",
                "done",
                "priority"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, priority)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        # sample using priority
        priorities = np.array([e.priority**ALPHA for e in self.memory])
        probs = priorities / priorities.sum()
        indices = np.random.choice(range(len(self.memory)), size=self.batch_size, p=probs)
        experiences = [self.memory[i] for i in indices]
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack(
            [e.action if e.action is not None else -1 for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones, probs[indices], indices)

    def update_priorities(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self.memory[i] = self.memory[i]._replace(priority=p.item())

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
