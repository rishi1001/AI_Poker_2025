from agents.agent import Agent
from gym_env import PokerEnv
import torch
import torch.nn as nn
import numpy as np
import random
from treys import Evaluator

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_action_types=5, num_raise_classes=100, num_discard_classes=3):
        """
        The network uses a shared base and three heads:
          - Action type head: outputs logits over 5 actions (FOLD, RAISE, CHECK, CALL, DISCARD)
          - Raise head: outputs logits over 100 values (to be encoded as percentages)
          - Discard head: outputs logits over 3 values (mapped to [-1, 0, 1])
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_type_head = nn.Linear(hidden_dim, num_action_types)
        self.raise_head = nn.Linear(hidden_dim, num_raise_classes)
        self.discard_head = nn.Linear(hidden_dim, num_discard_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_type_logits = self.action_type_head(x)
        raise_logits = self.raise_head(x)
        discard_logits = self.discard_head(x)
        return action_type_logits, raise_logits, discard_logits

class RLAgent(Agent):
    def __name__(self):
        return "RLAgent"

    def __init__(self, stream=False, temperature=1.0):
        """
        Initialize the RL agent with trained weights.
        
        Args:
            weights_path: Path to the saved model weights
            stream: Whether to stream logs
            temperature: Controls exploration (higher = more exploration)
        """
        super().__init__(stream)
        self.evaluator = Evaluator()
        self.temperature = temperature
        
        weights_path = 'sft_weights.pth'
        
        # Create policy network
        self.policy_net = PolicyNetwork(input_dim=13, hidden_dim=128)
        
        
        # Try to get the device the weights were saved on
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.policy_net.load_state_dict(torch.load(weights_path))
        else:
            self.device = torch.device("cpu")
            self.policy_net.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            
        self.policy_net.to(self.device)
        self.policy_net.eval()  # Set to evaluation mode
        self.logger.info(f"Successfully loaded weights from {weights_path}")
        
            
    def preprocess_observation(self, obs):
        """
        Converts the observation dictionary into a feature tensor.
        Same as in training code.
        """
        street = np.array([obs["street"] / 3.0])
        my_cards = np.array([(card + 1) / 28.0 for card in obs["my_cards"]])
        community_cards = np.array([(card + 1) / 28.0 for card in obs["community_cards"]])
        my_bet = np.array([obs["my_bet"] / 100.0])
        opp_bet = np.array([obs["opp_bet"] / 100.0])
        min_raise = np.array([obs["min_raise"] / 100.0])
        max_raise = np.array([obs["max_raise"] / 100.0])
        
        
        # Calculate equity using Monte Carlo simulation
        equity = self.calculate_equity(obs)
        
        features = np.concatenate([street, my_cards, community_cards, my_bet, opp_bet, min_raise, max_raise, equity])
        return features
    
    def calculate_equity(self, obs):
        """
        Calculate equity through Monte Carlo simulation, same as in the MC agent.
        """
        my_cards = [int(card) for card in obs["my_cards"]]
        community_cards = [card for card in obs["community_cards"] if card != -1]
        opp_discarded_card = [obs["opp_discarded_card"]] if obs["opp_discarded_card"] != -1 else []
        opp_drawn_card = [obs["opp_drawn_card"]] if obs["opp_drawn_card"] != -1 else []

        # Calculate equity through Monte Carlo simulation
        shown_cards = my_cards + community_cards + opp_discarded_card + opp_drawn_card
        non_shown_cards = [i for i in range(27) if i not in shown_cards]
        
        def evaluate_hand(cards):
            my_cards, opp_cards, community_cards = cards
            my_cards = list(map(int_to_card, my_cards))
            opp_cards = list(map(int_to_card, opp_cards))
            community_cards = list(map(int_to_card, community_cards))
            my_hand_rank = self.evaluator.evaluate(my_cards, community_cards)
            opp_hand_rank = self.evaluator.evaluate(opp_cards, community_cards)
            return my_hand_rank < opp_hand_rank

        # Run Monte Carlo simulation (200 simulations for faster inference)
        num_simulations = 200  # Less than in training for faster play
        wins = 0
        
        for _ in range(num_simulations):
            if len(non_shown_cards) >= (7 - len(community_cards) - len(opp_drawn_card)):
                drawn_cards = random.sample(non_shown_cards, 7 - len(community_cards) - len(opp_drawn_card))
                if evaluate_hand((my_cards, 
                                opp_drawn_card + drawn_cards[: 2 - len(opp_drawn_card)], 
                                community_cards + drawn_cards[2 - len(opp_drawn_card):])):
                    wins += 1
        
        equity = wins / num_simulations if num_simulations > 0 else 0.5
        return np.array([equity])

    def act(self, observation, reward, terminated, truncated, info):
        """
        Use the policy network to determine the action.
        """
        # Log game state for debugging
        if observation["street"] == 0:  # Preflop
            self.logger.debug(f"Hole cards: {[int_to_card(c) for c in observation['my_cards']]}")
        elif observation["community_cards"]:  # New community cards revealed
            visible_cards = [c for c in observation["community_cards"] if c != -1]
            if visible_cards:
                street_names = ["Preflop", "Flop", "Turn", "River"]
                self.logger.debug(f"{street_names[observation['street']]}: {[int_to_card(c) for c in visible_cards]}")

        # Preprocess observation to get feature tensor
        state_np = self.preprocess_observation(observation)
        state = torch.tensor(state_np, dtype=torch.float32).to(self.device)
        
        # Get valid actions and raise range
        valid_actions_tensor = torch.tensor(observation["valid_actions"], dtype=torch.float32).to(self.device)
        min_raise_val = observation["min_raise"]
        max_raise_val = observation["max_raise"]
        
        # Get logits from policy network
        with torch.no_grad():
            action_type_logits, raise_logits, discard_logits = self.policy_net(state)
        
        # Apply temperature for exploration/exploitation control
        if self.temperature != 1.0:
            action_type_logits = action_type_logits / self.temperature
            raise_logits = raise_logits / self.temperature
            discard_logits = discard_logits / self.temperature
        
        # Mask invalid actions
        mask = (valid_actions_tensor == 0)
        masked_logits = action_type_logits.clone()
        masked_logits[mask] = -1e9
        
        # Convert logits to probabilities
        action_type_probs = torch.softmax(masked_logits, dim=0)
        raise_probs = torch.softmax(raise_logits, dim=0)
        discard_probs = torch.softmax(discard_logits, dim=0)
        
        # Sample action
        action_type = torch.multinomial(action_type_probs, 1).item()
        raise_class = torch.multinomial(raise_probs, 1).item()
        discard_action = torch.multinomial(discard_probs, 1).item() - 1  # Adjust to -1, 0, 1
        
        # Convert raise class to actual raise amount using percentage encoding
        raise_percentage = raise_class / 99.0  # Convert to percentage between 0 and 1
        if min_raise_val == max_raise_val:
            raise_amount = min_raise_val
        else:
            raise_amount = min_raise_val + int(raise_percentage * (max_raise_val - min_raise_val))
        
        # Ensure raise amount is valid if raising
        if action_type == action_types.RAISE.value:
            raise_amount = max(min(raise_amount, max_raise_val), min_raise_val)
        else:
            raise_amount = 0
        
        # Handle discard action
        if action_type == action_types.DISCARD.value:
            if discard_action < 0:
                discard_action = 0  # Force valid index for discard
        else:
            discard_action = -1
        
        # Log decision and reasoning
        equity = state_np[-1]
        continue_cost = observation["opp_bet"] - observation["my_bet"]
        pot_size = observation["my_bet"] + observation["opp_bet"]
        pot_odds = continue_cost / (continue_cost + pot_size) if continue_cost > 0 else 0
        
        action_names = ["FOLD", "RAISE", "CHECK", "CALL", "DISCARD"]
        self.logger.debug(f"RL Decision: {action_names[action_type]}, Equity: {equity:.2f}, Pot odds: {pot_odds:.2f}")
        
        if action_type == action_types.RAISE.value and raise_amount > 20:
            self.logger.info(f"RL raising to {raise_amount} with equity {equity:.2f}")
        elif action_type == action_types.FOLD.value and observation["opp_bet"] > 20:
            self.logger.info(f"RL folding to bet of {observation['opp_bet']} with equity {equity:.2f}")
        elif action_type == action_types.DISCARD.value:
            my_cards = [int(card) for card in observation["my_cards"]]
            self.logger.debug(f"RL discarding card {discard_action}: {int_to_card(my_cards[discard_action])}")
        
        return action_type, raise_amount, discard_action

    def observe(self, observation, reward, terminated, truncated, info):
        """
        Called after each step to observe the result.
        """
        if terminated and abs(reward) > 20:
            self.logger.info(f"Hand completed with reward: {reward}")