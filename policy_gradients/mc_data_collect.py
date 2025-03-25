import os
import numpy as np
import random
import torch
import pickle
from tqdm import tqdm

# Import our environment and agents
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gym_env import PokerEnv
from agents.agent import Agent
# from agents.prob_agent import ProbabilityAgent  # Assuming you have this opponent
from agents.mc_agent import McAgent  # Your Monte Carlo agent

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
    
    # Placeholder for equity (will be filled in later)
    equity = np.array([0.5])
    
    features = np.concatenate([street, my_cards, community_cards, my_bet, opp_bet, min_raise, max_raise, equity])
    return features

def collect_monte_carlo_data(num_episodes=1000, opponent_type=McAgent, save_path="mc_training_data.pkl"):
    """
    Collects training data by having the Monte Carlo agent play against a specified opponent.
    Saves a dataset of (state, action_type, raise_amount, card_to_discard) tuples.
    """
    env = PokerEnv()
    mc_agent = McAgent(stream=False)  # The Monte Carlo agent
    opponent = opponent_type()  # Opponent agent
    
    dataset = []
    
    print(f"Collecting data from Monte Carlo agent vs {opponent.__name__()}")
    print(f"Planning to collect {num_episodes} episodes...")
    
    
    for episode in tqdm(range(num_episodes)):
        obs, info = env.reset()
        episode_done = False
        
        while not episode_done:
            acting_agent = obs[0]["acting_agent"]
            
            if acting_agent == 0:  # Monte Carlo agent's turn
                # We need to calculate the equity that the MC agent would calculate
                # This is needed to get the true decision-making process of the agent
                my_cards = [int(card) for card in obs[0]["my_cards"]]
                community_cards = [card for card in obs[0]["community_cards"] if card != -1]
                opp_discarded_card = [obs[0]["opp_discarded_card"]] if obs[0]["opp_discarded_card"] != -1 else []
                opp_drawn_card = [obs[0]["opp_drawn_card"]] if obs[0]["opp_drawn_card"] != -1 else []
                
                # Calculate equity through Monte Carlo simulation
                shown_cards = my_cards + community_cards + opp_discarded_card + opp_drawn_card
                non_shown_cards = [i for i in range(27) if i not in shown_cards]
                
                def evaluate_hand(cards):
                    my_cards, opp_cards, community_cards = cards
                    my_cards = list(map(PokerEnv.int_to_card, my_cards))
                    opp_cards = list(map(PokerEnv.int_to_card, opp_cards))
                    community_cards = list(map(PokerEnv.int_to_card, community_cards))
                    evaluator = mc_agent.evaluator
                    my_hand_rank = evaluator.evaluate(my_cards, community_cards)
                    opp_hand_rank = evaluator.evaluate(opp_cards, community_cards)
                    return my_hand_rank < opp_hand_rank
                
                # Run a smaller Monte Carlo simulation for data collection (faster)
                num_simulations = 200  # Reduced from 1000 for speed during data collection
                wins = 0
                for _ in range(num_simulations):
                    if len(non_shown_cards) >= (7 - len(community_cards) - len(opp_drawn_card)):
                        drawn_cards = random.sample(non_shown_cards, 7 - len(community_cards) - len(opp_drawn_card))
                        if evaluate_hand((my_cards, 
                                        opp_drawn_card + drawn_cards[: 2 - len(opp_drawn_card)], 
                                        community_cards + drawn_cards[2 - len(opp_drawn_card):])):
                            wins += 1
                
                equity = wins / num_simulations if num_simulations > 0 else 0.5
                
                # Now get the action from the Monte Carlo agent
                action_type, raise_amount, card_to_discard = mc_agent.act(
                    obs[0], reward=0, terminated=False, truncated=False, info={}
                )
                
                # Create feature vector with the calculated equity
                features = preprocess_observation(obs[0])
                features[-1] = equity  # Replace placeholder with actual equity
                
                # Record state and action
                data_point = {
                    "state": features,
                    "action_type": action_type,
                    "raise_amount": raise_amount,
                    "card_to_discard": card_to_discard,
                    "valid_actions": obs[0]["valid_actions"],
                    "min_raise": obs[0]["min_raise"],
                    "max_raise": obs[0]["max_raise"],
                    "equity": equity  # Store equity explicitly for analysis
                }
                dataset.append(data_point)
                
            else:  # Opponent's turn
                action_type, raise_amount, card_to_discard = opponent.act(
                    obs[1], reward=0, terminated=False, truncated=False, info={}
                )
            
            # Take action in environment
            action = (action_type, raise_amount, card_to_discard)
            obs, reward, episode_done, truncated, info = env.step(action)
            
            # If last step in episode, update MC agent (for logging purposes)
            if episode_done:
                mc_agent.observe(obs[0], reward[0], episode_done, truncated, info)
                
        # Print occasional progress updates
        if (episode + 1) % 100 == 0:
            print(f"Completed {episode + 1} episodes, collected {len(dataset)} data points")
    
    # Save the dataset
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    # Print some statistics about the collected data
    action_counts = {}
    for sample in dataset:
        action_type = sample["action_type"]
        if action_type not in action_counts:
            action_counts[action_type] = 0
        action_counts[action_type] += 1
    
    print("\nData collection statistics:")
    print(f"Total samples: {len(dataset)}")
    action_names = {0: "FOLD", 1: "RAISE", 2: "CHECK", 3: "CALL", 4: "DISCARD"}
    for action_type, count in action_counts.items():
        action_name = action_names.get(action_type, f"UNKNOWN({action_type})")
        percentage = count / len(dataset) * 100
        print(f"  {action_name}: {count} samples ({percentage:.1f}%)")
    
    print(f"\nData collection complete! Saved {len(dataset)} samples to {save_path}")
    return dataset

def analyze_dataset(dataset_path):
    """
    Analyze the collected dataset to understand the distribution of actions.
    """
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Dataset contains {len(dataset)} samples")
    
    # Count actions by type
    action_counts = {}
    for sample in dataset:
        action_type = sample["action_type"]
        if action_type not in action_counts:
            action_counts[action_type] = 0
        action_counts[action_type] += 1
    
    # Count actions by street
    street_counts = {}
    for sample in dataset:
        street = int(sample["state"][0] * 3)  # Convert back from normalized
        if street not in street_counts:
            street_counts[street] = 0
        street_counts[street] += 1
    
    # Print statistics
    print("\nAction distribution:")
    action_names = {0: "FOLD", 1: "RAISE", 2: "CHECK", 3: "CALL", 4: "DISCARD"}
    for action_type, count in sorted(action_counts.items()):
        action_name = action_names.get(action_type, f"UNKNOWN({action_type})")
        percentage = count / len(dataset) * 100
        print(f"  {action_name}: {count} samples ({percentage:.1f}%)")
    
    print("\nStreet distribution:")
    street_names = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River"}
    for street, count in sorted(street_counts.items()):
        street_name = street_names.get(street, f"UNKNOWN({street})")
        percentage = count / len(dataset) * 100
        print(f"  {street_name}: {count} samples ({percentage:.1f}%)")
    
    # Analyze equity distribution by action type
    print("\nEquity distribution by action type:")
    for action_type, name in action_names.items():
        if action_type in action_counts:
            equities = [sample["equity"] for sample in dataset if sample["action_type"] == action_type]
            if equities:
                avg_equity = sum(equities) / len(equities)
                min_equity = min(equities)
                max_equity = max(equities)
                print(f"  {name}: Avg equity = {avg_equity:.3f}, Min = {min_equity:.3f}, Max = {max_equity:.3f}")

if __name__ == "__main__":
    # Collect data against probability agent
    collect_monte_carlo_data(num_episodes=1000, save_path="mc_training_data.pkl")
    
    # Analyze the collected dataset
    analyze_dataset("mc_training_data.pkl")