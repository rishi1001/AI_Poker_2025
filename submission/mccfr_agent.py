import pickle
from agents.agent import Agent
from gym_env import PokerEnv
import random

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
        infoSet = str(observation)

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


    def get_action_for_info_set(self, infoSet, available_actions):
        # Check if the info set was visited during training
        # print("x", list(self.avg_strategy.keys())[0])
        # print("xx", infoSet)
        if infoSet in self.avg_strategy:
            # print(f"InfoSet {infoSet} was visited during training")
            strategy = self.avg_strategy[infoSet]
            # print(strategy)
        else:
            # Fallback to a uniform strategy
            strategy = {action: 1.0 / len(available_actions) for action in available_actions}

        # remove folds if fold prob < 50%
        new_strat = {}
        for a, p in strategy.items():
            if (a[0] != action_types.FOLD.value) or p > 0.5 and a[0] == action_types.FOLD.value:
                new_strat[a] = p
        # renormalize
        total = 0
        for a, p in new_strat.items():
            total += p
        for a, p in new_strat.items():
            new_strat[a] = p / total
        strategy = new_strat
        
        # Sample an action from the strategy
        r = random.random()
        cumulative_probability = 0.0
        for action, prob in strategy.items():
            cumulative_probability += prob
            if r < cumulative_probability:
                return action
        return available_actions[-1]  # fallback
    
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