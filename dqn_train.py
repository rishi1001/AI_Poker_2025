import argparse
import logging
import random
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import pandas as pd

from agents import test_agents
from rl import action_encoding, utils
from rl import state_encoding
from rl.one_player_gym_env import OnePlayerPokerEnv
from submission.player import PlayerAgent


# Define the Q-network as a simple feed-forward neural network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


# Experience replay buffer for storing transitions
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# Function to update the Q-network using a mini-batch from the replay buffer
def update(model, target_model, optimizer, replay_buffer, batch_size, gamma, logger):
    if len(replay_buffer) < batch_size:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Convert numpy arrays to torch tensors
    states = torch.FloatTensor(states)
    next_states = torch.FloatTensor(next_states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    dones = torch.FloatTensor(dones)

    # Compute current Q-values and the next Q-values from the target network
    q_values = model(states)
    next_q_values = target_model(next_states)

    # Gather Q-values for the chosen actions
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    # Calculate the expected Q-values using the max Q-value from next states
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = rewards + gamma * next_q_value * (1 - dones)

    # Compute loss and perform backpropagation
    # loss = nn.MSELoss()(q_value, expected_q_value.detach())
    # loss = F.smooth_l1_loss(q_value, expected_q_value.detach())
    loss = F.l1_loss(q_value, expected_q_value.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def main(load_from=None):
    # Hyperparameters and environment details (abstract example)
    state_dim = state_encoding.STATE_DIM  # Dimension of state (e.g., for a simple environment)
    action_dim = action_encoding.ACTION_DIM  # Number of possible actions
    num_hands = 1_000_000
    batch_size = 64
    gamma = 0.99
    replay_capacity = 10000
    target_update_freq = 50  # Update target network every 10 episodes
    save_every = 1000
    # opponent = PlayerAgent()
    opponent = test_agents.AllInAgent(stream=False)
    opponent.logger.disabled = True

    # Epsilon parameters for the epsilon-greedy policy
    epsilon_start = 1.0
    epsilon_final = 0.05
    epsilon_decay = 100

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Initialize environment, model, target network, optimizer, and replay buffer
    # (Replace `env` with your environment instance, e.g., from OpenAI Gym)
    env = OnePlayerPokerEnv(logger, bot=opponent)
    model = DQN(state_dim, action_dim)
    if load_from is not None:
        logger.info(f"Loading model from {load_from}")
        model.load_state_dict(torch.load(load_from))
    target_model = DQN(state_dim, action_dim)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=2 * 1e-3)
    replay_buffer = ReplayBuffer(replay_capacity)

    def epsilon_by_frame(frame_idx):
        return epsilon_final + (epsilon_start - epsilon_final) * np.exp(
            -1.0 * frame_idx / epsilon_decay
        )
    
    losses = []

    cumulative_reward = 0
    folds = 0
    raises = 0

    frame_idx = 0
    for hand_number in range(num_hands):
        small_blind_player = hand_number % 2
        state, reward, terminated, truncated, info = env.reset(
            options={"small_blind_player": small_blind_player, "hand_number": hand_number}
        )

        cumulative_reward += reward

        while not terminated:
            info["hand_number"] = hand_number
            assert state["acting_agent"] == 0

            epsilon = epsilon_by_frame(frame_idx)
            frame_idx += 1

            valid_actions_mask = action_encoding.compute_valid_actions_mask(
                state["valid_actions"], state["min_raise"], state["max_raise"]
            )

            state_tensor = state_encoding.obs_to_tensor(state)
            # Choose action using epsilon-greedy strategy
            if random.random() > epsilon:
                q_values = model(state_tensor)
                action_int = action_encoding.sample_action(q_values, valid_actions_mask)
                # q_values = q_values + valid_actions_mask
                # action_int = q_values.argmax()
            else:
                action_int = action_encoding.get_random_action(valid_actions_mask)

            action = action_encoding.action_int_to_action_tuple(action_int)
            if action_int == 0:
                folds += 1
            if action_int >= 5:
                raises += 1

            # Take action in the environment
            next_state, reward, terminated, truncated, info = env.step(action, hand_number)
            next_state_tensor = state_encoding.obs_to_tensor(next_state)
            # Store transition in replay buffer
            replay_buffer.push(state_tensor, action_int, reward, next_state_tensor, terminated)
            state = next_state

            # Update Q-network
            loss = update(model, target_model, optimizer, replay_buffer, batch_size, gamma, logger=logger)

            losses.append(loss)
            cumulative_reward += reward

        # Periodically update the target network
        if hand_number % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())
            # print(f"Episode {hand_number}: Updated target network")


        if hand_number % save_every == 0 and hand_number > 0:
            logger.info(f"Loss: {losses[-1]:.4f}\tHands: {hand_number}\tTotal reward: {cumulative_reward}\tReward/hands: {cumulative_reward/hand_number:.2f}\tFold ratio: {folds/hand_number:.2f}\tRaise ratio: {raises/hand_number:.2f}")
            # save in csv
            with open("loss_log.csv", "a") as f:
                f.write(f"{hand_number},{losses[-1]}\n")
            df = pd.read_csv("loss_log.csv", names=["train_step", "loss"])
            plt.plot(df["train_step"], df["loss"])
            plt.xlabel("Train Step")
            plt.ylabel("Loss")
            plt.yscale("log")
            plt.title("Loss Curve")
            plt.savefig("loss_curve.png")

            # save model
            torch.save(model.state_dict(), f"model_checkpoints/model_{hand_number//save_every//10}.pth")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--load_from", type=str, default=None)

    args = argument_parser.parse_args()
    load_from = args.load_from
    main(load_from)
