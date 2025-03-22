import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

from agents.agent import Agent
from gym_env import PokerEnv

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card


class PokerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=3):
        super(PokerNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, output_dim)  # Action type
        self.raise_head = nn.Linear(hidden_dim, 1)  # Raise amount (scaled)
        self.discard_head = nn.Linear(hidden_dim, 2)  # Card to discard (0 or 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_values = self.action_head(x)
        raise_value = self.raise_head(x)  # Raw value to be scaled
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
        # State representation size will depend on your observation space
        self.state_size = self._get_state_size()
        self.action_size = 3  # FOLD/CHECK, CALL, RAISE
        self.discard_size = 2  # Can discard card 0 or 1
        
        # Hyperparameters
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.target_update_freq = 100
        
        # Neural network and replay buffer
        self.memory = ReplayBuffer(capacity=10000)
        self.q_network = PokerNet(self.state_size, hidden_dim=128, output_dim=self.action_size)
        self.target_network = PokerNet(self.state_size, hidden_dim=128, output_dim=self.action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Tracking variables
        self.train_step = 0
        self.prev_state = None
        self.prev_action = None
        
    def _get_state_size(self):
        # Cards: 27 cards × 2 (my cards + community cards)
        # Game state: street, acting_agent, my_bet, opp_bet, 
        #             opp_discarded_card, opp_drawn_card, min_raise, max_raise
        # Valid actions: 1 for each action type
        return 27*2 + 8 + len(action_types)
    
    def _preprocess_state(self, observation):
        """Convert observation dict to flat state vector for the network"""
        # Extract relevant information from observation
        my_cards = observation["my_cards"]
        community_cards = observation.get("community_cards", [])
        street = observation["street"]
        acting_agent = observation["acting_agent"]
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        opp_discarded_card = observation["opp_discarded_card"]
        opp_drawn_card = observation["opp_drawn_card"]
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]
        valid_actions = observation["valid_actions"]
        
        # One-hot encode my cards (27 possible cards)
        my_cards_features = np.zeros(27)
        for card in my_cards:
            if card >= 0:  # Valid card
                my_cards_features[card] = 1
        
        # One-hot encode community cards
        community_cards_features = np.zeros(27)
        for card in community_cards:
            if card >= 0:  # Valid card
                community_cards_features[card] = 1
        
        # Game state features
        max_bet = 100.0  # Assuming maximum possible bet for normalization
        game_features = np.array([
            street / 3.0,  # Normalize street (0-3)
            acting_agent,  # 0 or 1
            my_bet / max_bet,  # Normalize bet
            opp_bet / max_bet,  # Normalize bet
            opp_discarded_card / 27.0 if opp_discarded_card >= 0 else -0.1,  # Normalize card index
            opp_drawn_card / 27.0 if opp_drawn_card >= 0 else -0.1,  # Normalize card index
            min_raise / max_bet,  # Normalize raise amount
            max_raise / max_bet,  # Normalize raise amount
        ])
        
        # Valid actions as features
        valid_actions_features = np.array(valid_actions, dtype=np.float32)
        
        # Concatenate all features
        state = np.concatenate([
            my_cards_features, 
            community_cards_features, 
            game_features, 
            valid_actions_features
        ])
        
        return state
    
    def _scale_raise_amount(self, raw_value, min_raise, max_raise):
        """Scale the network's raw raise value to a valid raise amount"""
        # Convert raw network output to a value between 0 and 1
        scaled = torch.sigmoid(raw_value).item()
        # Map to the range between min and max raise
        raise_amount = min_raise + scaled * (max_raise - min_raise)
        return int(raise_amount)
    
    def act(self, observation, reward, terminated, truncated, info):
        """Choose an action based on the current observation"""
        # Process observation to get state representation
        state = self._preprocess_state(observation)
        
        # Store previous experience if this isn't the first action
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
        
        # Get valid actions from environment
        valid_actions = observation["valid_actions"]
        
        # Decide whether to explore or exploit
        if random.random() < self.epsilon:
            # Explore: choose a random valid action
            valid_action_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
            action_type = random.choice(valid_action_indices)
            
            if action_type == action_types.RAISE.value:
                raise_amount = random.randint(observation["min_raise"], observation["max_raise"])
            else:
                raise_amount = 0
                
            if valid_actions[action_types.DISCARD.value]:
                card_to_discard = random.randint(0, 1)
            else:
                card_to_discard = -1
        else:
            # Exploit: use the model to predict best action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_values, raise_value, discard_values = self.q_network(state_tensor)
                
                # Mask invalid actions with large negative values
                masked_values = action_values.squeeze().numpy()
                for i, is_valid in enumerate(valid_actions):
                    if not is_valid and i < len(masked_values):
                        masked_values[i] = float('-inf')
                
                action_type = np.argmax(masked_values)
                
                if action_type == action_types.RAISE.value:
                    raise_amount = self._scale_raise_amount(
                        raise_value, 
                        observation["min_raise"], 
                        observation["max_raise"]
                    )
                else:
                    raise_amount = 0
                
                if valid_actions[action_types.DISCARD.value]:
                    card_to_discard = torch.argmax(discard_values).item()
                else:
                    card_to_discard = -1
        
        # Store current state and action for next step
        self.prev_state = state
        self.prev_action = (action_type, raise_amount, card_to_discard)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.logger.debug(f"DQN choosing: action={action_type}, raise={raise_amount}, discard={card_to_discard}")
        return action_type, raise_amount, card_to_discard
    
    def _train(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch of experiences
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones, valid_actions_list = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        # Unpack actions
        action_types, raise_amounts, discard_cards = zip(*actions)
        action_types = torch.LongTensor(action_types)
        raise_amounts = torch.FloatTensor(raise_amounts)
        discard_cards = torch.LongTensor([max(0, card) for card in discard_cards])  # -1 -> 0 for indexing
        
        # Get current Q values
        current_q_action, current_q_raise, current_q_discard = self.q_network(states)
        
        # Get Q values for chosen actions
        q_action = current_q_action.gather(1, action_types.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_action, next_q_raise, next_q_discard = self.target_network(next_states)
            
            # Apply mask for valid actions
            masked_next_q = []
            for i, valid_actions in enumerate(valid_actions_list):
                valid_indices = [j for j, is_valid in enumerate(valid_actions) if is_valid]
                if valid_indices:
                    max_q = max(next_q_action[i][j].item() for j in valid_indices)
                else:
                    max_q = 0  # No valid actions
                masked_next_q.append(max_q)
            
            next_q_values = torch.FloatTensor(masked_next_q)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(q_action, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def reset(self):
        """Reset agent state between episodes"""
        self.prev_state = None
        self.prev_action = None
    
    def save(self, path):
        """Save the model weights"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']