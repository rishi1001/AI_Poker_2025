import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import random
import time
from treys import Evaluator

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our poker environment
from gym_env import PokerEnv
# from agents.prob_agent import ProbabilityAgent
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
    
    # Calculate equity using Monte Carlo simulation (same as MC agent)
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

    def get_value(self, state):
        """
        Estimate the value of the given state using the value network.
        """
        if self.value_net is None:
            return 0
        with torch.no_grad():
            return self.value_net(state).item()

    def update_policy(self, trajectory, final_value=0):
        """
        Update the policy using the REINFORCE algorithm with a baseline.
        trajectory: list of (state, action, log_prob, entropy, reward) tuples
        final_value: estimated value of the final state (0 for terminal)
        """
        # Compute returns and advantages
        returns = []
        advantages = []
        R = final_value
        
        for _, _, _, _, reward in reversed(trajectory):
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        if self.use_baseline and trajectory:
            # Calculate advantages using the value network
            values = [self.value_net(state).item() for state, _, _, _, _ in trajectory]
            advantages = [ret - val for ret, val in zip(returns, values)]
            
            # Normalize advantages for more stable training
            if len(advantages) > 1:
                adv_mean = np.mean(advantages)
                adv_std = np.std(advantages)
                advantages = [(a - adv_mean) / (adv_std + 1e-8) for a in advantages]
        else:
            # Without baseline, advantage is just the return
            advantages = returns
            
            # Normalize returns for more stable training
            if len(returns) > 1:
                ret_mean = np.mean(returns)
                ret_std = np.std(returns)
                if ret_std > 1e-8:
                    advantages = [(r - ret_mean) / ret_std for r in returns]
        
        # Convert to tensors
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Policy loss with entropy regularization
        policy_loss = 0
        entropy_loss = 0
        
        for (state, _, log_prob, entropy, _), advantage in zip(trajectory, advantages):
            policy_loss += -log_prob * advantage
            entropy_loss += -entropy  # Negative because we want to maximize entropy
        
        # Add entropy regularization to encourage exploration
        policy_loss = policy_loss + self.entropy_beta * entropy_loss
        
        # Value loss (MSE between returns and predicted values)
        value_loss = 0
        if self.use_baseline and trajectory:
            for (state, _, _, _, _), ret in zip(trajectory, returns):
                value_pred = self.value_net(state)
                value_loss += (value_pred - ret) ** 2
            
            value_loss = value_loss / len(trajectory)  # Mean squared error
        
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
            'entropy': -entropy_loss.item() / len(trajectory) if trajectory else 0
        }

    def save_weights(self, path):
        """Save model weights."""
        torch.save(self.policy_net.state_dict(), path)
        if self.use_baseline:
            torch.save(self.value_net.state_dict(), path.replace('.pth', '_value.pth'))
        print(f"RL agent saved to {path}")

# --- Training Loop with Enhanced Monitoring and Evaluation ---

def train_agent(num_episodes=2000, 
                pretrained_path=None, 
                save_every=50, 
                eval_every=100,
                weight_path="rl_agent_weights.pth",
                use_baseline=True,
                learning_rate=1e-4):
    """
    Train the RL agent, optionally starting from pretrained weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = PokerEnv()
    agent = RLAgent(input_dim=13, 
                    hidden_dim=128, 
                    lr=learning_rate, 
                    pretrained_path=pretrained_path,
                    use_baseline=use_baseline)
    agent.policy_net.to(device)
    if use_baseline:
        agent.value_net.to(device)
    
    # Use ProbabilityAgent as the main opponent
    opponent_agent = McAgent()
    print(f"Using opponent: {opponent_agent.__name__()}")
    
    # Track performance
    all_rewards = []
    running_reward = 0
    best_avg_reward = -float('inf')
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        trajectory = []  # (state, action, log_prob, entropy, reward)
        episode_reward = 0
        done = False
        
        while not done:
            acting_agent = obs[0]["acting_agent"]
            
            if acting_agent == 0:  # RL agent's turn (player 0)
                # Preprocess and convert to tensor
                state_np = preprocess_observation(obs[0])
                state = torch.tensor(state_np, dtype=torch.float32).to(device)
                
                valid_actions_tensor = torch.tensor(obs[0]["valid_actions"], dtype=torch.float32).to(device)
                min_raise_val = obs[0]["min_raise"]
                max_raise_val = obs[0]["max_raise"]
                
                action, log_prob, entropy = agent.select_action(state, valid_actions_tensor, min_raise_val, max_raise_val)
                our_turn = True
            else:  # Opponent's turn (player 1)
                action = opponent_agent.act(obs[1], reward=0, terminated=False, truncated=False, info={})
                our_turn = False
                log_prob = None
                entropy = None
            
            # Take action in environment
            next_obs, reward, done, truncated, info = env.step(action)
            r = reward[0]  # RL agent's reward is at index 0
            episode_reward += r
            
            if our_turn:
                trajectory.append((state, action, log_prob, entropy, r))
            
            obs = next_obs
        
        # Update policy and track performance
        if trajectory:
            loss_dict = agent.update_policy(trajectory)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}, Loss info: {loss_dict}")
        
        all_rewards.append(episode_reward)
        
        # Update running reward
        running_reward = 0.05 * episode_reward + 0.95 * (running_reward if episode > 0 else episode_reward)
        
        # Print status
        if (episode + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode+1}, Reward: {episode_reward:.2f}, Running Reward: {running_reward:.2f}, Time: {elapsed:.1f}s")
            start_time = time.time()
        
        # Save weights periodically
        if (episode + 1) % save_every == 0:
            agent.save_weights(weight_path)
            print(f"Saved weights to {weight_path}")
        
        # Evaluate agent periodically
        if (episode + 1) % eval_every == 0:
            avg_reward = evaluate_agent(agent, env, opponent_agent, num_eval_episodes=50, device=device)
            print(f"Evaluation after {episode+1} episodes: Average Reward = {avg_reward:.2f}")
            
            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_weights(f"best_{weight_path}")
                print(f"New best model with average reward: {avg_reward:.2f}")
    
    # Final weight save
    agent.save_weights(weight_path)
    print(f"Final weights saved to {weight_path}")
    
    # Return training statistics
    return {
        "rewards": all_rewards,
        "best_avg_reward": best_avg_reward
    }

def evaluate_agent(agent, env, opponent, num_eval_episodes=50, device=None):
    """
    Evaluate the agent's performance over multiple episodes.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    total_reward = 0
    
    for _ in range(num_eval_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            acting_agent = obs[0]["acting_agent"]
            
            if acting_agent == 0:  # RL agent's turn
                with torch.no_grad():  # Disable gradients for faster evaluation
                    state_np = preprocess_observation(obs[0])
                    state = torch.tensor(state_np, dtype=torch.float32).to(device)
                    valid_actions_tensor = torch.tensor(obs[0]["valid_actions"], dtype=torch.float32).to(device)
                    min_raise_val = obs[0]["min_raise"]
                    max_raise_val = obs[0]["max_raise"]
                    
                    action, _, _ = agent.select_action(state, valid_actions_tensor, min_raise_val, max_raise_val)
            else:  # Opponent's turn
                action = opponent.act(obs[1], reward=0, terminated=False, truncated=False, info={})
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward[0]  # RL agent's reward
            
        total_reward += episode_reward
    
    return total_reward / num_eval_episodes

if __name__ == "__main__":
    # Example usage
    train_agent(
        num_episodes=2000, 
        pretrained_path="sft_weights.pth",
        weight_path="rl_agent_weights.pth",
        use_baseline=True,  # Set to True to use a value network as baseline
        learning_rate=1e-4
    )