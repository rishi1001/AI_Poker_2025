import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt  # For plotting loss curves
import os

from agents.agent import Agent
from gym_env import PokerEnv

action_types = PokerEnv.ActionType

def encode_card(card):
    """
    Encodes a card as a 12-dimensional vector:
      - 9 dimensions for rank (2, 3, ..., 9, Ace)
      - 3 dimensions for suit (diamonds, hearts, spades)
    If card is invalid (e.g. -1), returns a zero vector.
    """
    rank_vector = np.zeros(9, dtype=np.float32)
    suit_vector = np.zeros(3, dtype=np.float32)
    if card >= 0:
        rank_index = card % 9
        suit_index = card // 9
        rank_vector[rank_index] = 1.0
        suit_vector[suit_index] = 1.0
    return np.concatenate([rank_vector, suit_vector])

class PokerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=4):
        """
        Now output_dim=4 corresponds to our four decisions:
          0: CHECK (or FOLD if CHECK isn’t valid)
          1: CALL
          2: RAISE
          3: DISCARD
        """
        super(PokerNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, output_dim)  # For action type selection
        self.raise_head = nn.Linear(hidden_dim, 1)            # For scaled raise amount
        self.discard_head = nn.Linear(hidden_dim, 2)          # For card to discard (0 or 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_values = self.action_head(x)
        raise_value = self.raise_head(x)
        discard_values = self.discard_head(x)
        return action_values, raise_value, discard_values

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done, valid_actions):
        self.buffer.append((state, action, reward, next_state, done, valid_actions))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNPokerAgent(Agent):
    def __name__(self):
        return "DQNPokerAgent"
    
    def __init__(self, player_idx):
        super().__init__(player_idx)
        # New state representation:
        # - Hole cards: 2 cards * 12 features = 24
        # - Community cards: fixed 5 cards * 12 features = 60
        # - Game state features: street, acting_agent, my_bet, opp_bet, opp_discarded_card, opp_drawn_card, min_raise, max_raise = 8
        # - Valid actions: one binary feature per environment action (len(action_types))
        # We subtract 1 because the action type also contains an INVALID action.
        self.state_size = self._get_state_size()
        # Now we use 4 actions in our network:
        # 0: CHECK (or FOLD if check not valid), 1: CALL, 2: RAISE, 3: DISCARD
        self.action_size = 4  
        self.discard_size = 2  # Discard card index: 0 or 1
        
        # Hyperparameters
        self.gamma = 0.95         # Discount factor
        self.epsilon = 1.0        # Exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.target_update_freq = 100
        
        # Initialize networks and replay buffer
        self.memory = ReplayBuffer(capacity=10000)
        self.q_network = PokerNet(self.state_size, hidden_dim=128, output_dim=self.action_size)
        self.target_network = PokerNet(self.state_size, hidden_dim=128, output_dim=self.action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # For storing previous state/action pair and loss tracking
        self.train_step = 0
        self.prev_state = None
        self.prev_action = None
        self.losses = []  # Store loss values for plotting
        
    def _get_state_size(self):
        # Hole cards: 2 * 12 = 24
        # Community cards: 5 * 12 = 60
        # Game state features: 8
        # Valid actions: len(action_types) - 1 (ignoring INVALID)
        return 24 + 60 + 8 + len(action_types) - 1
    
    def _preprocess_state(self, observation):
        """Convert observation dict into a flat state vector using the new card encoding."""
        # Encode hole cards (always 2 cards)
        my_cards = observation["my_cards"]
        my_cards_features = np.concatenate([encode_card(card) for card in my_cards])
        
        # Encode community cards; pad to fixed length (5 cards)
        max_community_cards = 5
        community_cards = observation.get("community_cards", [])
        community_cards_features_list = [encode_card(card) for card in community_cards]
        while len(community_cards_features_list) < max_community_cards:
            community_cards_features_list.append(np.zeros(12, dtype=np.float32))
        community_cards_features = np.concatenate(community_cards_features_list)
        
        # Game state features
        max_bet = 100.0  # for normalization
        opp_discarded = observation["opp_discarded_card"]
        opp_drawn = observation["opp_drawn_card"]
        
        # TODO: You might want to one-hot encode some of these features in the future.
        game_features = np.array([
            observation["street"] / 3.0,                     # Street normalized
            float(observation["acting_agent"]),              # Acting agent (0 or 1)
            observation["my_bet"] / max_bet,                 # My bet normalized
            observation["opp_bet"] / max_bet,                # Opponent's bet normalized
            (opp_discarded / 27.0) if opp_discarded >= 0 else -0.1,  # Opponent discarded card (normalized)
            (opp_drawn / 27.0) if opp_drawn >= 0 else -0.1,          # Opponent drawn card (normalized)
            observation["min_raise"] / max_bet,              # Min raise normalized
            observation["max_raise"] / max_bet               # Max raise normalized
        ], dtype=np.float32)
        
        # Valid actions vector (one-hot for environment actions)
        valid_actions_features = np.array(observation["valid_actions"], dtype=np.float32)
        
        # Concatenate all features into a single state vector
        state = np.concatenate([
            my_cards_features, 
            community_cards_features, 
            game_features, 
            valid_actions_features
        ])
        
        return state
    
    def _scale_raise_amount(self, raw_value, min_raise, max_raise):
        """Scale the network's raw raise value output to a valid raise amount."""
        scaled = torch.sigmoid(raw_value).item()  # Value in (0,1)
        raise_amount = min_raise + scaled * (max_raise - min_raise)
        return int(raise_amount)
    
    def act(self, observation, reward, terminated, truncated, info):
        """
        Decide an action based on the current observation.
        We first determine the network's choice among our four actions:
          0: CHECK (or FOLD if check isn't valid),
          1: CALL,
          2: RAISE,
          3: DISCARD.
        Then we map that to the actual environment action.
        """
        state = self._preprocess_state(observation)
        
        # Store previous experience if available.
        if self.prev_state is not None and self.prev_action is not None:
            done = terminated or truncated
            self.memory.add(
                self.prev_state, 
                self.prev_action, 
                reward, 
                state, 
                done,
                observation["valid_actions"]
            )
            
            # Train the network
            self._train()
        
        # Define our mapping from network output index to environment action:
        # For index 0, we prefer CHECK if valid; otherwise FOLD.
        mapping = {
            0: (action_types.CHECK.value if observation["valid_actions"][action_types.CHECK.value] else action_types.FOLD.value),
            1: action_types.CALL.value,
            2: action_types.RAISE.value,
            3: action_types.DISCARD.value
        }
        
        # Build valid network action indices based on our mapping.
        valid_network_actions = []
        for net_act in range(self.action_size):
            env_act = mapping[net_act]
            if observation["valid_actions"][env_act]:
                valid_network_actions.append(net_act)
                
        if random.random() < self.epsilon:
            # Exploration: choose a random valid network action.
            if valid_network_actions:
                chosen_net_action = random.choice(valid_network_actions)
            else:
                # Fallback: if none valid, choose a default action (e.g., fold)
                chosen_net_action = 0  
        else:
            # Exploitation: use the model's prediction.
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_values, raise_value, discard_values = self.q_network(state_tensor)
                action_values = action_values.squeeze().numpy()
                # Mask out actions that are not valid (according to our mapping)
                for net_act in range(self.action_size):
                    env_act = mapping[net_act]
                    if not observation["valid_actions"][env_act]:
                        action_values[net_act] = float('-inf')
                chosen_net_action = int(np.argmax(action_values))
        
        # Map the chosen network action to the actual environment action type.
        chosen_env_action = mapping[chosen_net_action]
        
        # Initialize raise_amount and card_to_discard.
        raise_amount = 0
        card_to_discard = -1
        
        if chosen_env_action == action_types.RAISE.value:
            if random.random() < self.epsilon:  # Exploration: choose random valid raise.
                raise_amount = random.randint(observation["min_raise"], observation["max_raise"])
            else:
                with torch.no_grad():
                    _, raw_raise, _ = self.q_network(torch.FloatTensor(state).unsqueeze(0))
                    raise_amount = self._scale_raise_amount(raw_raise, observation["min_raise"], observation["max_raise"])
        elif chosen_env_action == action_types.DISCARD.value:
            # Only decide on a card to discard if DISCARD is chosen.
            if random.random() < self.epsilon:
                card_to_discard = random.randint(0, 1)
            else:
                with torch.no_grad():
                    _, _, discard_values = self.q_network(torch.FloatTensor(state).unsqueeze(0))
                    card_to_discard = int(torch.argmax(discard_values).item())
        
        # Store state and action for later training.
        self.prev_state = state
        # Save the full action tuple (env action, raise, discard)
        self.prev_action = (chosen_env_action, raise_amount, card_to_discard)
        
        # Decay epsilon.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.logger.info(f"DQN choosing: action={chosen_env_action}, raise={raise_amount}, discard={card_to_discard}")
        return chosen_env_action, raise_amount, card_to_discard
    
    def _train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones, valid_actions_list = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        # Unpack actions: each is a tuple (env_action, raise_amount, discard)
        action_env_types, raise_amounts, discard_cards = zip(*actions)
        # Map environment action to network action index:
        #   CALL -> net index 1, RAISE -> net index 2, DISCARD -> net index 3, else (CHECK/FOLD) -> net index 0.
        net_actions = []
        for env_act in action_env_types:
            if env_act == action_types.CALL.value:
                net_actions.append(1)
            elif env_act == action_types.RAISE.value:
                net_actions.append(2)
            elif env_act == action_types.DISCARD.value:
                net_actions.append(3)
            else:
                net_actions.append(0)
                
        net_actions = torch.LongTensor(net_actions)
        raise_amounts = torch.FloatTensor(raise_amounts)
        discard_cards = torch.LongTensor([max(0, card) for card in discard_cards])  # Convert -1 to 0 for indexing
        
        current_q_action, current_q_raise, current_q_discard = self.q_network(states)
        q_action = current_q_action.gather(1, net_actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_action, next_q_raise, next_q_discard = self.target_network(next_states)
            masked_next_q = []
            for i, valid_actions in enumerate(valid_actions_list):
                # Reconstruct valid network indices from the environment's valid actions using our mapping.
                valid_net_indices = []
                mapping = {
                    0: (action_types.CHECK.value if valid_actions[action_types.CHECK.value] else action_types.FOLD.value),
                    1: action_types.CALL.value,
                    2: action_types.RAISE.value,
                    3: action_types.DISCARD.value
                }
                for net_act in range(self.action_size):
                    if valid_actions[mapping[net_act]]:
                        valid_net_indices.append(net_act)
                if valid_net_indices:
                    max_q = max(next_q_action[i][j].item() for j in valid_net_indices)
                else:
                    max_q = 0
                masked_next_q.append(max_q)
            next_q_values = torch.FloatTensor(masked_next_q)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = F.smooth_l1_loss(q_action, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Log the loss value and store it for plotting.
        loss_val = loss.item()
        self.losses.append(loss_val)
        self.logger.info(f"Train step {self.train_step}, Loss: {loss_val:.4f}")
    
        with open("loss_log.csv", "a") as f:
            f.write(f"{self.train_step},{loss_val}\n")
        
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Save model periodically
        if self.train_step % 100 == 0:
            self.save("dqn_poker_agent.pth")
            self.logger.info(f"Model saved at step {self.train_step}")
    
    def reset(self):
        """Reset agent state between episodes."""
        self.prev_state = None
        self.prev_action = None
    
    def save(self, path):
        """Save model weights."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Load model weights."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
