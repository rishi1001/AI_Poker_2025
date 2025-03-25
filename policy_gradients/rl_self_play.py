import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import random
import time
from treys import Evaluator
import copy
import sys
# Import our poker environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gym_env import PokerEnv
from agents.mc_agent import McAgent

# Define preprocessing functions (same as in data collection)
def preprocess_observation(obs):
    """
    Converts the observation dictionary for one player into a feature tensor.
    Features include:
      - street (normalized by 3)
      - my_cards (2 values, shifted and normalized)
      - community_cards (5 values, shifted and normalized)
      - my_bet and opp_bet (normalized by 100)
      - min_raise and max_raise (normalized by 100)
      - computed equity (a scalar between 0 and 1)
    """
    street = np.array([obs["street"] / 3.0])
    my_cards = np.array([(card + 1) / 28.0 for card in obs["my_cards"]])
    community_cards = np.array([(card + 1) / 28.0 for card in obs["community_cards"]])
    my_bet = np.array([obs["my_bet"] / 100.0])
    opp_bet = np.array([obs["opp_bet"] / 100.0])
    min_raise = np.array([obs["min_raise"] / 100.0])
    max_raise = np.array([obs["max_raise"] / 100.0])
    
    # Calculate equity using Monte Carlo simulation
    equity = calculate_equity(obs)
    
    features = np.concatenate([street, my_cards, community_cards, my_bet, opp_bet, min_raise, max_raise, equity])
    return features

def calculate_equity(obs):
    """
    Calculate equity through Monte Carlo simulation, exactly as in the Monte Carlo agent.
    """
    my_cards = [int(card) for card in obs["my_cards"]]
    community_cards = [card for card in obs["community_cards"] if card != -1]
    opp_discarded_card = [obs["opp_discarded_card"]] if obs["opp_discarded_card"] != -1 else []
    opp_drawn_card = [obs["opp_drawn_card"]] if obs["opp_drawn_card"] != -1 else []

    # Calculate equity through Monte Carlo simulation
    shown_cards = my_cards + community_cards + opp_discarded_card + opp_drawn_card
    non_shown_cards = [i for i in range(27) if i not in shown_cards]
    
    # Create an evaluator instance
    evaluator = Evaluator()
    
    def evaluate_hand(cards):
        my_cards, opp_cards, community_cards = cards
        my_cards = list(map(PokerEnv.int_to_card, my_cards))
        opp_cards = list(map(PokerEnv.int_to_card, opp_cards))
        community_cards = list(map(PokerEnv.int_to_card, community_cards))
        my_hand_rank = evaluator.evaluate(my_cards, community_cards)
        opp_hand_rank = evaluator.evaluate(opp_cards, community_cards)
        return my_hand_rank < opp_hand_rank

    # Run Monte Carlo simulation (use a smaller number for faster RL)
    num_simulations = 1000  # Reduced from 1000 for speed during RL
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

# --- Define the Policy Network ---
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

# --- Define the RL Agent using REINFORCE with Baseline ---
class RLAgent:
    def __init__(self, input_dim, hidden_dim=128, lr=1e-4, pretrained_path=None, use_baseline=True):
        self.policy_net = PolicyNetwork(input_dim, hidden_dim)
        
        # Value network for baseline (to reduce variance)
        if use_baseline:
            self.value_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.optimizer = torch.optim.Adam([
                {'params': self.policy_net.parameters(), 'lr': lr},
                {'params': self.value_net.parameters(), 'lr': lr}
            ])
        else:
            self.value_net = None
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Load pretrained weights if available
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            self.policy_net.load_state_dict(torch.load(pretrained_path))
        else:
            print("Starting with random weights")
            
        self.gamma = 0.99  # Discount factor
        self.use_baseline = use_baseline
        self.entropy_beta = 0.01  # Entropy regularization coefficient
        
        # For tracking performance
        self.episode_rewards = []
        self.running_reward = 0
        self.best_reward = -float('inf')
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        if self.use_baseline:
            self.value_net.to(self.device)

    def select_action(self, state, valid_actions, min_raise, max_raise):
        """
        Given the state tensor and valid actions, sample an action tuple.
        Returns:
          (action_type, raise_amount, card_to_discard), log_prob, entropy
        """
        action_type_logits, raise_logits, discard_logits = self.policy_net(state)
        
        # Mask invalid actions before sampling action type
        mask = (valid_actions == 0)
        masked_logits = action_type_logits.clone()
        masked_logits[mask] = -1e9

        action_type_dist = torch.distributions.Categorical(logits=masked_logits)
        raise_dist = torch.distributions.Categorical(logits=raise_logits)
        discard_dist = torch.distributions.Categorical(logits=discard_logits)

        # Sample actions
        action_type = action_type_dist.sample()
        raise_class = raise_dist.sample()
        discard_action = discard_dist.sample()

        # Calculate log probabilities
        log_prob = (action_type_dist.log_prob(action_type) +
                    raise_dist.log_prob(raise_class) +
                    discard_dist.log_prob(discard_action))
        
        # Calculate entropy for regularization
        entropy = action_type_dist.entropy() + raise_dist.entropy() + discard_dist.entropy()

        # Convert to Python values
        action_type = action_type.item()
        
        # Convert the raise_class (0-99) to an actual raise amount between min_raise and max_raise
        raise_percentage = raise_class.item() / 99.0  # Convert to percentage between 0 and 1
        
        # Handle the case where min_raise equals max_raise
        if min_raise == max_raise:
            raise_amount = min_raise
        else:
            raise_amount = min_raise + int(raise_percentage * (max_raise - min_raise))
        
        # Ensure raise amount is valid
        if action_type == PokerEnv.ActionType.RAISE.value:
            raise_amount = int(max(min(raise_amount, max_raise), min_raise))
        else:
            raise_amount = 0

        # Map discard output: 0 -> -1, 1 -> 0, 2 -> 1
        discard_action = discard_action.item() - 1
        if action_type == PokerEnv.ActionType.DISCARD.value:
            if discard_action < 0:
                discard_action = 0  # Force a valid index (0 or 1)
        else:
            discard_action = -1

        return (action_type, raise_amount, discard_action), log_prob, entropy

    def act(self, obs, reward=0, terminated=False, truncated=False, info={}):
        """
        Interface method to make the agent compatible with the PokerEnv.
        This allows the agent to be used as either player.
        """
        with torch.no_grad():
            state_np = preprocess_observation(obs)
            state = torch.tensor(state_np, dtype=torch.float32).to(self.device)
            valid_actions_tensor = torch.tensor(obs["valid_actions"], dtype=torch.float32).to(self.device)
            min_raise_val = obs["min_raise"]
            max_raise_val = obs["max_raise"]
            
            action, _, _ = self.select_action(state, valid_actions_tensor, min_raise_val, max_raise_val)
            return action

    def get_value(self, state):
        """
        Estimate the value of the given state using the value network.
        """
        if self.value_net is None:
            return 0
        with torch.no_grad():
            return self.value_net(state).item()

    def update_policy(self, trajectories):
        """
        Update the policy using the REINFORCE algorithm with a baseline.
        trajectories: Dictionary with player indices as keys and lists of 
                     (state, action, log_prob, entropy, reward) tuples as values
        """
        # Process all trajectories for both players
        all_states = []
        all_log_probs = []
        all_entropies = []
        all_returns = []
        
        for player, trajectory in trajectories.items():
            if not trajectory:
                continue
                
            # Compute returns for this player
            returns = []
            R = 0
            for _, _, _, _, reward in reversed(trajectory):
                R = reward + self.gamma * R
                returns.insert(0, R)
            
            # Extract states, log_probs, and entropies
            states = [state for state, _, _, _, _ in trajectory]
            log_probs = [log_prob for _, _, log_prob, _, _ in trajectory]
            entropies = [entropy for _, _, _, entropy, _ in trajectory]
            
            all_states.extend(states)
            all_log_probs.extend(log_probs)
            all_entropies.extend(entropies)
            all_returns.extend(returns)
        
        if not all_returns:  # No valid trajectories
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}
            
        # Convert to tensors
        all_returns = torch.tensor(all_returns, dtype=torch.float32).to(self.device)
        
        # Calculate advantages
        advantages = all_returns.clone()
        if self.use_baseline:
            # Calculate values using the value network
            values = torch.cat([self.value_net(state) for state in all_states])
            advantages = all_returns - values.detach()
            
            # Normalize advantages for more stable training
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss with entropy regularization
        policy_loss = 0
        entropy_loss = 0
        
        for log_prob, entropy, advantage in zip(all_log_probs, all_entropies, advantages):
            policy_loss += -log_prob * advantage
            entropy_loss += -entropy  # Negative because we want to maximize entropy
        
        # Add entropy regularization to encourage exploration
        policy_loss = policy_loss + self.entropy_beta * entropy_loss
        
        # Value loss (MSE between returns and predicted values)
        value_loss = 0
        if self.use_baseline and all_states:
            values = torch.cat([self.value_net(state) for state in all_states])
            value_loss = torch.nn.functional.mse_loss(values.squeeze(), all_returns)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        if self.use_baseline:
            torch.nn.utils.clip_grad_norm_(list(self.policy_net.parameters()) + 
                                          list(self.value_net.parameters()), 1.0)
        else:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item() if self.use_baseline else 0,
            'entropy': -entropy_loss.item() / len(all_entropies) if all_entropies else 0
        }

    def save_weights(self, path):
        """Save model weights."""
        torch.save(self.policy_net.state_dict(), path)
        if self.use_baseline:
            torch.save(self.value_net.state_dict(), path.replace('.pth', '_value.pth'))
        print(f"RL agent saved to {path}")
        
    def clone(self):
        """Create a clone of this agent with the same weights."""
        clone = RLAgent(input_dim=13, hidden_dim=128, 
                        use_baseline=self.use_baseline)
        clone.policy_net.load_state_dict(self.policy_net.state_dict())
        if self.use_baseline and self.value_net is not None:
            clone.value_net.load_state_dict(self.value_net.state_dict())
        return clone

# --- Training Loop with Self-Play ---

def train_with_self_play(num_episodes=2000, 
                         pretrained_path=None, 
                         save_every=50, 
                         eval_every=100,
                         weight_path="self_play_rl_weights.pth",
                         use_baseline=True,
                         learning_rate=1e-4,
                         opponent_update_freq=100):
    """
    Train the RL agent through self-play, optionally starting from pretrained weights.
    
    Parameters:
    - num_episodes: Number of episodes to train for
    - pretrained_path: Path to pretrained weights from SFT
    - save_every: How often to save weights
    - eval_every: How often to evaluate against a fixed opponent
    - weight_path: Where to save the weights
    - use_baseline: Whether to use a value network baseline
    - learning_rate: Learning rate for the optimizer
    - opponent_update_freq: How often to update the opponent's weights
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = PokerEnv()
    
    # Create main agent
    main_agent = RLAgent(input_dim=13, 
                         hidden_dim=128, 
                         lr=learning_rate, 
                         pretrained_path=pretrained_path,
                         use_baseline=use_baseline)
    
    # Create opponent agent (initially a clone of the main agent)
    opponent_agent = main_agent.clone()
    
    # Create a fixed opponent for evaluation
    eval_opponent = McAgent()
    
    # Track performance
    all_rewards = []
    running_reward = 0
    best_avg_reward = -float('inf')
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        
        # Trajectories for both players
        trajectories = {0: [], 1: []}
        
        # Rewards for this episode
        episode_rewards = {0: 0, 1: 0}
        
        done = False
        
        while not done:
            acting_agent = obs[0]["acting_agent"]
            
            # Preprocess for the current player
            state_np = preprocess_observation(obs[acting_agent])
            state = torch.tensor(state_np, dtype=torch.float32).to(device)
            
            valid_actions_tensor = torch.tensor(obs[acting_agent]["valid_actions"], dtype=torch.float32).to(device)
            min_raise_val = obs[acting_agent]["min_raise"]
            max_raise_val = obs[acting_agent]["max_raise"]
            
            # Get action from appropriate agent
            if acting_agent == 0:  # Main agent's turn
                action, log_prob, entropy = main_agent.select_action(
                    state, valid_actions_tensor, min_raise_val, max_raise_val
                )
            else:  # Opponent agent's turn
                action, log_prob, entropy = opponent_agent.select_action(
                    state, valid_actions_tensor, min_raise_val, max_raise_val
                )
            
            # Take action in environment
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Record state, action, log_prob, entropy, reward for the acting agent
            trajectories[acting_agent].append((state, action, log_prob, entropy, reward[acting_agent]))
            
            # Update rewards
            episode_rewards[0] += reward[0]
            episode_rewards[1] += reward[1]
            
            obs = next_obs
        
        # Update policy for main agent using both trajectories
        loss_dict = main_agent.update_policy(trajectories)
        
        # Track performance
        all_rewards.append(episode_rewards[0])  # Track main agent's rewards
        
        # Update running reward
        running_reward = 0.05 * episode_rewards[0] + 0.95 * (running_reward if episode > 0 else episode_rewards[0])
        
        # Update opponent agent periodically (uses the main agent's weights)
        if (episode + 1) % opponent_update_freq == 0:
            opponent_agent = main_agent.clone()
            print(f"Updated opponent agent at episode {episode+1}")
        
        # Print status
        if (episode + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode+1}, Reward: {episode_rewards[0]:.2f}, "
                  f"Running Reward: {running_reward:.2f}, Time: {elapsed:.1f}s")
            print(f"Loss info: {loss_dict}")
            start_time = time.time()
        
        # Save weights periodically
        if (episode + 1) % save_every == 0:
            main_agent.save_weights(weight_path)
            print(f"Saved weights to {weight_path}")
        
        # Evaluate agent periodically against a fixed opponent
        if (episode + 1) % eval_every == 0:
            avg_reward = evaluate_agent(main_agent, env, eval_opponent, num_eval_episodes=50)
            print(f"Evaluation after {episode+1} episodes: Average Reward = {avg_reward:.2f}")
            
            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                main_agent.save_weights(f"best_{weight_path}")
                print(f"New best model with average reward: {avg_reward:.2f}")
    
    # Final weight save
    main_agent.save_weights(weight_path)
    print(f"Final weights saved to {weight_path}")
    
    # Return training statistics
    return {
        "rewards": all_rewards,
        "best_avg_reward": best_avg_reward
    }

def evaluate_agent(agent, env, opponent, num_eval_episodes=50):
    """
    Evaluate the agent's performance over multiple episodes.
    """
    total_reward = 0
    
    for _ in range(num_eval_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            acting_agent = obs[0]["acting_agent"]
            
            if acting_agent == 0:  # RL agent's turn
                action = agent.act(obs[0])
            else:  # Opponent's turn
                action = opponent.act(obs[1], reward=0, terminated=False, truncated=False, info={})
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward[0]  # RL agent's reward
            
        total_reward += episode_reward
    
    return total_reward / num_eval_episodes

if __name__ == "__main__":
    # Example usage
    train_with_self_play(
        num_episodes=2000, 
        pretrained_path="sft_weights.pth",  # Start from the SFT weights
        weight_path="self_play_rl_weights.pth",
        use_baseline=True,
        learning_rate=1e-4,
        opponent_update_freq=100  # Update opponent every 100 episodes
    )