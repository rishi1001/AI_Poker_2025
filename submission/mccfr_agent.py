import pickle
from agents.agent import Agent
from gym_env import PokerEnv
import random
import math

action_types = PokerEnv.ActionType


class MCCFRAgent(Agent):
    def __name__(self):
        return "MCCFRAgent"

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        # Initialize any instance variables here
        self.hand_number = 0
        self.last_action = None
        self.won_hands = 0

        file_name = "avg_strategy_street_0.pkl"
        
        print(f"Loading avg_strategy from {file_name}")
        # Load avg_strategy from the pickle file
        with open(file_name, "rb") as f:
            self.avg_strategy = pickle.load(f)
        print(f"Loaded avg_strategy from {file_name}")

    def compute_information_set(obs):
        """
            obs = {
                "street": state["street"],
                "acting_agent": state["acting_agent"],
                "my_cards": state["player_cards"][player],
                "community_cards": state["community_cards"][:num_cards_to_reveal] + [-1] * (5 - num_cards_to_reveal),
                "my_bet": state["bets"][player],
                "opp_bet": state["bets"][1 - player],
                "opp_discarded_card": state["discarded_cards"][1 - player],
                "opp_drawn_card": state["drawn_cards"][1 - player],
                "my_discarded_card": state["discarded_cards"][player],
                "my_drawn_card": state["drawn_cards"][player],
                "min_raise": state["min_raise"],
                "max_raise": self.max_player_bet - max(state["bets"]),
                "valid_actions": self._get_valid_actions(state, player),
            }
        """
        flop_cards_sorted = sorted(obs["community_cards"][:3])
        turn_card = obs["community_cards"][3]
        river_card = obs["community_cards"][4]
        player = obs["acting_agent"]
        my_cards_sorted = sorted(obs["my_cards"])
        community_cards_sorted = "-".join(flop_cards_sorted + [turn_card, river_card])
        my_bet_binned = int(math.log2(obs["my_bet"])) if obs["my_bet"] > 0 else 0
        opp_bet_binned = int(math.log2(obs["opp_bet"])) if obs["opp_bet"] > 0 else 0
        min_raise_binned = int(math.log2(obs["min_raise"])) if obs["min_raise"] > 0 else 0
        max_raise_binned = int(math.log2(obs["max_raise"])) if obs["max_raise"] > 0 else 0
        valid_actions = "".join(obs["valid_actions"])

        # TODO have the exact cards discarded and drawn later
        i_discarded = obs["my_discarded_card"] != -1
        opp_discarded = obs["opp_discarded_card"] != -1

        return f"{player}_{my_cards_sorted}_{community_cards_sorted}_{my_bet_binned}_{opp_bet_binned}_{min_raise_binned}_{max_raise_binned}_{valid_actions}_{i_discarded}_{opp_discarded}"

    def act(self, observation, reward, terminated, truncated, info):
        if "my_discarded_card" not in observation and "my_drawn_card" not in observation:
            observation["my_discarded_card"] = -1
            observation["my_drawn_card"] = -1
        observation = {
            "street": observation["street"],
            "acting_agent": observation["acting_agent"],
            "my_cards": observation["my_cards"],
            "community_cards": observation["community_cards"],
            "my_bet": observation["my_bet"],
            "opp_bet": observation["opp_bet"],
            "opp_discarded_card": observation["opp_discarded_card"],
            "opp_drawn_card": observation["opp_drawn_card"],
            "my_discarded_card": observation["my_discarded_card"],
            "my_drawn_card": observation["my_drawn_card"],
            "min_raise": observation["min_raise"],
            "max_raise": observation["max_raise"],
            "valid_actions": observation["valid_actions"],
        }
        # print("hi from mccfr agent")
        # First, get the list of valid actions we can take
        valid_actions = observation["valid_actions"]

        actual_valid_actions = self.get_valid_actions(valid_actions, observation)

        # Get the info set for this observation
        infoSet = self.compute_information_set(observation)

        # print(f"valid_actions: {actual_valid_actions} - infoSet: {infoSet} - in strat {infoSet in self.avg_strategy}")

        # Get the available actions for this info set
        sampled_action = self.get_action_for_info_set(infoSet, actual_valid_actions)

        # print(f"sampled_action: {sampled_action}")

        return sampled_action



    def observe(self, observation, reward, terminated, truncated, info):
        # Log interesting events when observing opponent's actions
        pass
        if terminated:
            # self.logger.info(f"Game ended with reward: {reward}")
            self.hand_number += 1
            if reward > 0:
                self.won_hands += 1
            self.last_action = None
        else:
            # log observation keys
            # self.logger.info(f"Observation keys: {observation}")
            pass
        

    def find_nearest_info_set(self, info_set, available_actions):
        """
        Find a nearby information set by tweaking specific parameters.
        This function only modifies bet sizes and raise parameters, keeping
        everything else constant for efficiency.
        """
        if info_set in self.avg_strategy:
            return info_set
        
        # Parse the info set
        parts = info_set.split('_')
        if len(parts) < 8:
            return None  # Malformed info set
        
        player = parts[0]
        cards = parts[1]
        community = parts[2]
        my_bet = int(parts[3])
        opp_bet = int(parts[4])
        min_raise = int(parts[5])
        max_raise = int(parts[6])
        valid_actions = parts[7]
        i_discarded = parts[8] if len(parts) > 8 else "-1"
        opp_discarded = parts[9] if len(parts) > 9 else "-1"
        
        # Try modifying only the betting parameters
        for my_bet_adj in range(max(0, my_bet-2), my_bet+3):
            for opp_bet_adj in range(max(0, opp_bet-2), opp_bet+3):
                for min_raise_adj in range(max(0, min_raise-2), min_raise+3):
                    for max_raise_adj in range(max(0, max_raise-2), max_raise+3):
                        # Create a new info set with adjusted parameters
                        new_info_set = f"{player}_{cards}_{community}_{my_bet_adj}_{opp_bet_adj}_{min_raise_adj}_{max_raise_adj}_{valid_actions}_{i_discarded}_{opp_discarded}"
                        
                        if new_info_set in self.avg_strategy:
                            # Check if this strategy has actions compatible with our available actions
                            strategy = self.avg_strategy[new_info_set]
                            available_action_types = {action[0] for action in available_actions}
                            
                            if any(action[0] in available_action_types for action in strategy):
                                return new_info_set
        
        # If tweaking bet sizes doesn't work, try looking for any info set with same cards and valid actions
        # prefix = f"{player}_{cards}_{community}_"
        # suffix = f"_{valid_actions}_{i_discarded}_{opp_discarded}"
        
        # for key in self.avg_strategy:
        #     if key.startswith(prefix) and key.endswith(suffix):
        #         strategy = self.avg_strategy[key]
        #         available_action_types = {action[0] for action in available_actions}
                
        #         if any(action[0] in available_action_types for action in strategy):
        #             return key
        
        return None

    def get_action_for_info_set(self, info_set, available_actions):
        # First check if we have this exact info set
        if info_set in self.avg_strategy:
            strategy = self.avg_strategy[info_set]
        else:
            # Try to find a nearest neighbor
            nearest_info_set = self.find_nearest_info_set(info_set, available_actions)
            
            if nearest_info_set and nearest_info_set in self.avg_strategy:
                strategy = self.avg_strategy[nearest_info_set]
            else:
                # Fallback to a uniform strategy if no good neighbor found
                strategy = {action: 1.0 / len(available_actions) for action in available_actions}

        # Filter strategy to only include available actions
        available_action_tuples = [tuple(action) for action in available_actions]
        filtered_strategy = {}
        for action, prob in strategy.items():
            action_tuple = tuple(action)
            if action_tuple in available_action_tuples:
                filtered_strategy[action] = prob
        
        # If no matching actions, use uniform strategy
        if not filtered_strategy:
            filtered_strategy = {tuple(action): 1.0 / len(available_actions) for action in available_actions}
        
        # Remove folds if fold probability < 50%
        new_strat = {}
        for a, p in filtered_strategy.items():
            if (a[0] != action_types.FOLD.value) or (p > 0.5 and a[0] == action_types.FOLD.value):
                new_strat[a] = p
                
        # If we filtered out all actions, restore them
        if not new_strat:
            new_strat = filtered_strategy
            
        # Renormalize
        total = sum(new_strat.values())
        if total > 0:
            for a, p in new_strat.items():
                new_strat[a] = p / total
        else:
            # If all probabilities were zero, use uniform
            for a in new_strat:
                new_strat[a] = 1.0 / len(new_strat)
        
        # Sample an action from the strategy
        r = random.random()
        cumulative_probability = 0.0
        for action, prob in new_strat.items():
            cumulative_probability += prob
            if r < cumulative_probability:
                return action
                
        # Fallback
        return available_actions[-1]
    
    def get_valid_actions(self, valid, obs):
        acting = obs["acting_agent"]
        actions = []
        # FOLD
        if valid[action_types.FOLD.value] and obs["my_bet"] < 80 and obs["street"] == 0 and False:
            actions.append((action_types.FOLD.value, 0, -1))
        # CHECK
        if valid[action_types.CHECK.value]:
            actions.append((action_types.CHECK.value, 0, -1))
        # CALL
        if valid[action_types.CALL.value]:
            actions.append((action_types.CALL.value, 0, -1))
        # DISCARD (two options: discard card 0 or 1)
        if valid[action_types.DISCARD.value] and obs["my_bet"] < 80 and obs["street"] == 0:
            actions.append((action_types.DISCARD.value, 0, 0))
            actions.append((action_types.DISCARD.value, 0, 1))
        # RAISE: include options for min, max, and if possible intermediate values.
        if valid[action_types.RAISE.value]:
            min_raise = obs["min_raise"]
            max_raise = obs["max_raise"]
            pot = obs["my_bet"] + obs["opp_bet"]
            actions.append((action_types.RAISE.value, min_raise, -1))
            actions.append((action_types.RAISE.value, max_raise, -1))
            if min_raise < pot < max_raise:
                actions.append((action_types.RAISE.value, pot, -1))
            if min_raise < pot // 2 < max_raise:
                actions.append((action_types.RAISE.value, pot // 2, -1))
        return actions