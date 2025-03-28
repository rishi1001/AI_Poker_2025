import pickle
import numpy as np
import random
from treys import Evaluator, Card
import sys
# sys.path.append("submission")
from gym_env import PokerEnv, WrappedEval
int_to_card = PokerEnv.int_to_card

# Load the strategy file (adjust path as needed)
def load_strategy(filename="submission/merged_avg_strategy.pkl"):
    with open(filename, "rb") as f:
        avg_strategy = pickle.load(f)
    return avg_strategy

# Card definitions
# EIGHT_SPADE = 6    # 8♠ (0*9 + 6)
# EIGHT_DIAMOND = 24 # 8♦ (2*9 + 6)


# ACE_SPADE = 8      # A♠ (0*9 + 8)
# ACE_DIAMOND = 26   # A♦ (2*9 + 8)

# Function to compute info set
def compute_info_set(my_cards, community_cards=[], opp_discarded_card=[], opp_drawn_card=[], 
                    my_bet=1, opp_bet=2, valid_actions="11101"):

    breakpoint()
    # Calculate equity through Monte Carlo simulation like in the original code
    shown_cards = my_cards + (community_cards or []) + \
                  ([opp_discarded_card] if opp_discarded_card != -1 else []) + \
                  ([opp_drawn_card] if opp_drawn_card != -1 else [])
    
    
    
    # Cards that are not shown
    non_shown_cards = [i for i in range(27) if i not in shown_cards]
    
    # Define evaluation function (equivalent to evaluate_hand in original code)
    evaluator = WrappedEval()
    
    def evaluate_hand(cards):
            my_cards, opp_cards, community_cards = cards
            my_cards = list(map(int_to_card, my_cards))
            opp_cards = list(map(int_to_card, opp_cards))
            community_cards = list(map(int_to_card, community_cards))
            my_hand_rank = evaluator.evaluate(my_cards, community_cards)
            opp_hand_rank = evaluator.evaluate(opp_cards, community_cards)
            return my_hand_rank < opp_hand_rank

        # Run Monte Carlo simulation
    num_simulations = 5000
    wins = sum(
        evaluate_hand((my_cards, opp_drawn_card + drawn_cards[: 2 - len(opp_drawn_card)], community_cards + drawn_cards[2 - len(opp_drawn_card) :]))
        for _ in range(num_simulations)
        if (drawn_cards := random.sample(non_shown_cards, 7 - len(community_cards) - len(opp_drawn_card)))
    )
    equity = wins / num_simulations
    binned_equity = int(equity * 8)
    breakpoint()
    
    
    
    
    # No flush potential yet (pre-flop with different suits)
    flush_number = 0
    
    # Valid actions mapping
    valid_actions_map = {
        "11101": 0, # FOLD, CHECK, CALL, RAISE
        "10010": 1, # FOLD, DISCARD
        "11011": 2, # FOLD, CHECK, CALL, DISCARD
        "10101": 3, # FOLD, CALL, RAISE
        "11010": 4, # FOLD, CHECK, DISCARD
        "11100": 5, # FOLD, CHECK, CALL
        "10100": 6, # FOLD, CALL
        "10011": 7, # FOLD, DISCARD, RAISE
    }
    valid_actions_number = valid_actions_map.get(valid_actions, 0)
    
    # Calculate pot odds
    continue_cost = opp_bet - my_bet
    pot_size = my_bet + opp_bet
    pot_odds = continue_cost / pot_size if continue_cost > 0 else 0
    binned_pot_odds = int(pot_odds * 5)
    
    return (binned_equity, valid_actions_number, binned_pot_odds)

# Convert info set to integer
def info_set_to_integer(info_set):
    radices = [9, 8, 5]
    return encode_fields(info_set, radices)

def encode_fields(values, radices):
    result = 0
    for i in range(len(values)):
        result = result * radices[i] + values[i]
    return result

# Interpret action distribution
def interpret_distribution(distribution):
    if distribution is None:
        return "No strategy found for this info set"
    
    actions = [
        "FOLD",
        "CHECK",
        "CALL",
        "DISCARD lower card",
        "DISCARD higher card",
        "RAISE minimum",
        "RAISE maximum",
        "RAISE 3x pot",
        "RAISE pot"
    ]
    
    # Sort by probability
    action_probs = [(actions[i], dist) for i, dist in enumerate(distribution)]
    action_probs.sort(key=lambda x: x[1], reverse=True)
    
    result = "Action distribution:\n"
    for action, prob in action_probs:
        if prob > 0:
            result += f"{action}: {prob:.2%}\n"
    
    return result

# Main analysis function
def analyze_hand(avg_strategy):
    my_cards = [EIGHT_SPADE, EIGHT_DIAMOND]
    # my_cards = [ACE_SPADE, ACE_DIAMOND]
    # opp_cards = [EIGHT_HEART, NINE_HEART]
    
    
    # Create a mock observation similar to what the agent receives
    observation = {
        "my_cards": my_cards,
        "community_cards": [-1, -1, -1, -1, -1],  # No community cards pre-flop
        "street": 0,  # Pre-flop
        "valid_actions": [1, 1, 1, 0, 1],  # FOLD, CHECK, CALL, DISCARD, RAISE
        "my_bet": 1,
        "opp_bet": 2,
        "min_raise": 3,
        "max_raise": 100,
        "opp_discarded_card": -1,
        "opp_drawn_card": -1
    }
    
    # Compute info set for pre-flop position
    info_set = compute_info_set(my_cards)
    info_set_integer = info_set_to_integer(info_set)
    
    print(f"Hand: 8♠, 8♦ vs 8♥, 9♥ (pre-flop)")
    print(f"Info set: {info_set}")
    print(f"Info set integer: {info_set_integer}")
    breakpoint()
    
    # Get action distribution from strategy
    if info_set_integer < len(avg_strategy):
        distribution = avg_strategy[info_set_integer]
        if distribution is not None:
            print(interpret_distribution(distribution))
            
            # Sample action from distribution
            sampled_action = np.random.choice(len(distribution), p=distribution)
            actions = ["FOLD", "CHECK", "CALL", "DISCARD lower", "DISCARD higher", 
                      "RAISE min", "RAISE max", "RAISE 3x pot", "RAISE pot"]
            print(f"Sampled action: {actions[sampled_action]}")
            
            # Convert to actual action tuple
            action_tuple = action_int_to_action(sampled_action, observation)
            print(f"Action tuple: {action_tuple}")
            
            # Apply safety checks
            safety_action = safety_check(action_tuple, info_set, observation)
            if safety_action != action_tuple:
                print(f"Safety override applied! New action: {safety_action}")
        else:
            print("No strategy defined for this info set - would fall back to ChallengeOne")
            print("ChallengeOne would likely CALL since equity > pot odds")
    else:
        print("Info set index out of bounds in strategy table")

# Convert action_int to action tuple (from original code)
def action_int_to_action(action_int, observation):
    # Define action types enum-like values
    class ActionType:
        FOLD = 0
        CHECK = 1
        CALL = 2
        DISCARD = 3
        RAISE = 4
    
    my_hand = observation["my_cards"]
    max_raise = observation["max_raise"]
    min_raise = min(observation["min_raise"], max_raise)
    my_bet = observation["my_bet"]
    opp_bet = observation["opp_bet"]
    
    if action_int == 0:
        return (ActionType.FOLD, 0, -1)
    elif action_int == 1:
        return (ActionType.CHECK, 0, -1)
    elif action_int == 2:
        return (ActionType.CALL, 0, -1)
    elif action_int == 3:  # Discard lower card
        lower_card_idx = 0 if my_hand[0] % 9 <= my_hand[1] % 9 else 1
        return (ActionType.DISCARD, 0, lower_card_idx)
    elif action_int == 4:  # Discard higher card
        higher_card_idx = 0 if my_hand[0] % 9 >= my_hand[1] % 9 else 1
        return (ActionType.DISCARD, 0, higher_card_idx)
    elif action_int == 5:  # Raise (min_raise)
        min_raise = min(min_raise, max_raise)
        return (ActionType.RAISE, min_raise, -1)
    elif action_int == 6:  # Raise (max_raise)
        return (ActionType.RAISE, max_raise, -1)
    elif action_int == 7:  # Raise (pot)
        pot = my_bet + opp_bet
        mult = pot * 3
        safe_bet = max(min_raise, min(max_raise, mult))
        return (ActionType.RAISE, safe_bet, -1)
    elif action_int == 8:  # Raise (half pot)
        pot = my_bet + opp_bet
        safe_bet = max(min_raise, min(max_raise, pot))
        return (ActionType.RAISE, safe_bet, -1)
    else:
        raise ValueError(f"Invalid action integer: {action_int}")

# Safety check function (from original code)
def safety_check(action, info_set, observation):
    class ActionType:
        FOLD = 0
        CHECK = 1
        CALL = 2
        DISCARD = 3
        RAISE = 4
        
    binned_equity, valid_actions_number, binned_pot_odds = info_set
    
    # Convert action tuple to a more readable format for printing
    action_names = ["FOLD", "CHECK", "CALL", "DISCARD", "RAISE"]
    current_action = action_names[action[0]]
    
    if binned_equity >= 7:
        if action[0] != ActionType.RAISE:
            if observation["valid_actions"][ActionType.RAISE] == 1:
                print(f"Safety check: big binned equity and we were going to {current_action} - raising 3*pot instead")
                return (ActionType.RAISE, max(observation["min_raise"], 
                                         min(observation["max_raise"], 
                                            (observation["my_bet"] + observation["opp_bet"]) * 3)), -1)
            if observation["valid_actions"][ActionType.CALL] == 1:
                return (ActionType.CALL, 0, -1)
            if observation["valid_actions"][ActionType.CHECK] == 1:
                return (ActionType.CHECK, 0, -1)
                
    if action[0] == ActionType.FOLD:
        if observation["valid_actions"][ActionType.CHECK] == 1:
            print("Safety check: CHECK available but agent was going to FOLD")
            return (ActionType.CHECK, 0, -1)
        if binned_equity >= 5:
            if observation["valid_actions"][ActionType.CALL] == 1:
                print("Safety check: Good equity, CALL instead of FOLD")
                return (ActionType.CALL, 0, -1)
            elif binned_equity >= 7:
                print("Safety check: Very good equity, RAISE instead of FOLD")
                return (ActionType.RAISE, observation["min_raise"], -1)
                
    if action[0] == ActionType.DISCARD:
        if observation["valid_actions"][ActionType.CHECK] == 1:
            print(f"Safety check: DISCARDING on {binned_equity}, {valid_actions_number}, {binned_pot_odds} - CHECK was possible")
            return (ActionType.CHECK, 0, -1)
            
    if action[0] == ActionType.RAISE:
        if binned_equity < 6:
            print(f"Safety check: Raising with low equity {binned_equity}, {valid_actions_number}, {binned_pot_odds}")
            if observation["valid_actions"][ActionType.CHECK] == 1:
                return (ActionType.CHECK, 0, -1)
            elif observation["valid_actions"][ActionType.CALL] == 1:
                return (ActionType.CALL, 0, -1)
            else:
                return (ActionType.RAISE, observation["min_raise"], -1)
                
    return action

if __name__ == "__main__":
    try:
        avg_strategy = load_strategy()
        analyze_hand(avg_strategy)
    except Exception as e:
        print(f"Error: {e}")
        print("Continuing with a simple analysis...")
        
        # Create a mock strategy for demonstration
        mock_avg_strategy = [None] * 1000
        mock_distribution = [0.10, 0.25, 0.40, 0.0, 0.0, 0.15, 0.05, 0.05, 0.0]
        mock_avg_strategy[161] = mock_distribution  # Use the expected info_set_integer
        
        print("Using mock strategy for demonstration:")
        analyze_hand(mock_avg_strategy)