from model import QNetwork
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from replay_buffers import ReplayBuffer

# Check CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():

    def __init__(self, state_size, action_size, gamma, tau, lr, update_every, buffer_size, batch_size):

        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every

        self.buffer_size = buffer_size
        self.batch_size = batch_size

        print("Agent created!")
        self.state_size = state_size
        self.action_size = action_size

        # Define the Q-Network (local & target)
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)

        # Define the optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size)

        # Initialize time step
        self.timestep = 0

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy."""

        # Take state
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # Set the local network to evaluation mode
        self.qnetwork_local.eval()

        # Disable gradient calculation (to reduce memory usage)
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        # Set the local network back to training mode
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn once every "self.update_every" time steps
        self.timestep = (self.timestep + 1) % self.update_every

        if self.timestep == 0:
            # If we have enough samples experienced, get a subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)  # Call the learn function

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples."""

        # Extract from experiences
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Update targets
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_locals = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_locals, Q_targets)

        # Minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft-update target model
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
         θ_target = τ*θ_local + (1 - τ)*θ_target"""

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
