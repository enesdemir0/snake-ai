import numpy as np
from neural_network import NeuralNetwork
import random
from collections import deque


class Agent:
    def __init__(self, state_size=34, action_size=3, hidden_size=128, learning_rate=0.001,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, gamma=0.9,
                 replay_buffer_size=5000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        self.model = NeuralNetwork(state_size, hidden_size, action_size, learning_rate)

        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state = np.array(state)
            if len(state) != self.state_size:
                raise ValueError(f"Expected input size {self.state_size}, but got {len(state)}")
            q_values = self.model.run(state).flatten()
            return np.argmax(q_values)

    def reset_epsilon(self, value=0.1):
        self.epsilon = value

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self, state=None, action=None, reward=None, next_state=None, done=None):
        # Store current experience if provided
        if state is not None:
            self.store_experience(state, action, reward, next_state, done)

        if len(self.replay_buffer) < self.batch_size:
            # Not enough samples to train yet
            return

        # Sample random batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)

        for state_b, action_b, reward_b, next_state_b, done_b in batch:
            state_b = np.array(state_b)
            next_state_b = np.array(next_state_b)

            q_values = self.model.run(state_b).flatten()
            q_values_next = self.model.run(next_state_b).flatten()

            target = q_values.copy()
            target[action_b] = reward_b if done_b else reward_b + self.gamma * np.max(q_values_next)

            self.model.train(state_b, target)

        # Decay epsilon after batch training
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        self.model.save(filename + "_nn.npz")
        np.savez(filename + "_agent.npz", epsilon=self.epsilon)

    def load(self, filename):
        try:
            self.model.load(filename + "_nn.npz")
            data = np.load(filename + "_agent.npz")
            self.epsilon = data['epsilon'].item()
            print(f"Loaded model and epsilon={self.epsilon:.4f}")
        except FileNotFoundError:
            print("No saved model found, starting fresh.")
