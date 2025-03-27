import pickle
import time

import numpy as np
import torch
from agents.agent import Agent
from gym_env import PokerEnv
import random
import math

import gym_env

action_types = PokerEnv.ActionType


class PlayerAgent(Agent):
    def __name__(self):
        return "MCCFRAgent"

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        # Initialize any instance variables here
        self.hand_number = 0
        self.last_action = None
        self.won_hands = 0
        self.about_to_begin = True
        self.i_am_small_blind = True
        self.last_street_bet = 0
        self.last_street = 0
        self.cumulative_profit = 0

        self.chal_one = ChallengeOne()

        file_name = "submission/merged_strategy_100000000.pkl"
        
        print(f"Loading avg_strategy from {file_name}")
        # Load avg_strategy from the pickle file
        with open(file_name, "rb") as f:
            self.avg_strategy = pickle.load(f)
        print(f"Loaded avg_strategy from {file_name}")

    def act(self, observation, reward, terminated, truncated, info):
        if self.cumulative_profit > 0.5 + (1000-self.hand_number) * 1.5:
            return (ActionType.FOLD.value, 0, -1)
        # state tracking
        if self.about_to_begin:
            self.about_to_begin = False
            self.i_am_small_blind = observation["my_bet"] == 1
        street = observation["street"]
        if street != self.last_street:
            self.last_street = street
            self.last_street_bet = observation["my_bet"]
        # end of state tracking
        assert("my_discarded_card" in observation)

        if observation["street"] > 0 or observation["my_bet"] > 2 or observation["opp_bet"] > 2:
            return self.chal_one.act(observation, reward, terminated, truncated, info)

        valid_actions = observation["valid_actions"]

        infoSet = self.compute_information_set_reduced(observation)
        actual_valid_actions = self.get_valid_actions_str(valid_actions, observation)

        sampled_action_str = self.get_action_str_for_info_set(infoSet, actual_valid_actions, observation)
        sampled_action_tuple = action_str_to_action_tuple(sampled_action_str, observation)

        return sampled_action_tuple


    def observe(self, observation, reward, terminated, truncated, info):
        # Log interesting events when observing opponent's actions
        # state tracking
        if self.about_to_begin:
            self.about_to_begin = False
            self.i_am_small_blind = observation["my_bet"] == 1
        # end of state tracking
        if terminated:
            self.about_to_begin = True
            # self.logger.info(f"Game ended with reward: {reward}")
            self.hand_number += 1
            if reward > 0:
                self.won_hands += 1
            self.last_action = None
            self.cumulative_profit += reward
        else:
            # log observation keys
            # self.logger.info(f"Observation keys: {observation}")
            pass
        

    def get_action_str_for_info_set(self, info_set, available_actions_str, obs):
        # post all in simplification
        if ("RAISE" not in available_actions_str) and "FOLD" in available_actions_str and "CHECK" in available_actions_str:
            return "CHECK"
        # First check if we have this exact info set
        if info_set in self.avg_strategy:
            strategy = self.avg_strategy[info_set]
        else:
            # # Try to find a nearest neighbor
            # nearest_info_set = self.find_nearest_info_set(info_set, available_actions)
            
            # if nearest_info_set and nearest_info_set in self.avg_strategy:
            #     strategy = self.avg_strategy[nearest_info_set]
            # else:
            #     # # Fallback to a uniform strategy if no good neighbor found
            #     # strategy = {action: 1.0 / len(available_actions) for action in available_actions}
            strategy = self.subgame_solve(obs)

        # Filter strategy to only include available actions
        new_strat = {}
        for action_str, prob in strategy.items():
            if action_str in available_actions_str:
                new_strat[action_str] = prob
            else:
                raise ValueError(f"Action {action_str} not in available actions {available_actions_str}")

        # Just sample by taking the action with the high probability
        # max_a = None
        # max_p = 0
        # for action_str, prob in new_strat.items():
        #     if prob > max_p:
        #         max_p = prob
        #         max_a = action_str
        # return max_a
        
        # Sample an action from the strategy
        r = random.random()
        cumulative_probability = 0.0
        for action_str, prob in new_strat.items():
            cumulative_probability += prob
            if r < cumulative_probability:
                return action_str
                
        # Fallback
        return available_actions_str[-1]
    
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

    def get_valid_actions_str(self, valid, obs):
        acting = obs["acting_agent"]
        actions = []
        if valid[action_types.FOLD.value]:
            actions.append("FOLD")
        if valid[action_types.CHECK.value]:
            actions.append("CHECK")
        if valid[action_types.CALL.value]:
            actions.append("CALL")
        if valid[action_types.DISCARD.value]:
            actions.append("DISCARD_0")
            actions.append("DISCARD_1")
        if valid[action_types.RAISE.value]:
            actions.append("RAISE_MIN")
            actions.append("RAISE_MAX")
            actions.append("RAISE_POT")
            actions.append("RAISE_HALF_POT")
        return actions
    
    def subgame_solve(self, obs):
        info_set = self.compute_information_set_reduced(obs)
        generated_opp_hands = self.generate_all_opp_hands(obs)
        info_set_strategy_sum = dict()
        game = PokerGame()
        trainer = MCCFRTrainer(game)
        for opp_hand in generated_opp_hands:
            current_state = self.get_game_state(obs, opp_hand)
            trainer.reset()
            _, strategy_sums = trainer.train_strategy_sum(1, custom_initial_state=current_state)

            info_set_sum = strategy_sums[info_set]
            for action in info_set_sum:
                if action not in info_set_strategy_sum:
                    info_set_strategy_sum[action] = 0
                info_set_strategy_sum[action] += info_set_sum[action]

        strategy = {}
        total = sum(info_set_strategy_sum.values())
        for action, prob in info_set_strategy_sum.items():
            strategy[action] = prob / total
        return strategy
    
    def generate_all_opp_hands(self, obs):
        # MAKE SURE THEY ARE SORTED TO AVOID DOUBLES
        # generate all cards opp could have
        # depends on whether they drew or not
        # if they drew, then we only have one card to guess
        # if they did not draw, then we have two cards to guess
        deck = set(list(np.arange(27)))
        cards_to_remove = set(obs["my_cards"] + obs["community_cards"] + [obs["my_discarded_card"], obs["opp_discarded_card"], obs["my_drawn_card"], obs["opp_drawn_card"]])
        cards_to_remove.discard(-1)
        deck -= cards_to_remove
        deck = list(deck)
        result = []
        if obs["opp_drawn_card"] == -1:
            # generate all 2 cards combinations from the remaining cards in the deck
            for card1 in deck:
                for card2 in deck:
                    if card1 < card2: # avoid duplicates
                        result.append([card1, card2])
        else:
            # generate all 1 card combinations from the remaining cards in the deck
            # concatenate with the drawn card
            first_card = obs["opp_drawn_card"]
            for card in deck:
                result.append([first_card, card])
        
        return result
    
    def get_game_state(self, obs, opp_hand):
        seed = torch.randint(0, 1_000_000_000, (1,)).item()
        rng = np.random.RandomState(seed)
        # Create deck: our game uses 27 cards (integers 0..26)
        deck = set(list(np.arange(27)))
        # TODO: remove our cards and the assumed opponent's cards from the deck, and the discarded/drawn cards
        cards_to_remove = set(obs["my_cards"] + opp_hand + obs["community_cards"] + [obs["my_discarded_card"], obs["opp_discarded_card"], obs["my_drawn_card"], obs["opp_drawn_card"]])
        cards_to_remove.discard(-1)
        deck -= cards_to_remove
        deck = list(deck)
        rng.shuffle(deck)
        player_cards = [obs["my_cards"], opp_hand]
        bets = [obs["my_bet"], obs["opp_bet"]]
        discarded_cards = [obs["my_discarded_card"], obs["opp_discarded_card"]]
        drawn_cards = [obs["my_drawn_card"], obs["opp_drawn_card"]]
        if not self.i_am_small_blind:
            player_cards = [player_cards[1], player_cards[0]]
            bets = [bets[1], bets[0]]
            discarded_cards = [discarded_cards[1], discarded_cards[0]]
            drawn_cards = [drawn_cards[1], drawn_cards[0]]
        # Set blinds (by default player 0 is small blind and player 1 is big blind)
        state = {
            "seed": seed,
            "deck": deck,  # remaining deck
            "street": obs["street"],
            "bets": bets,
            "discarded_cards": discarded_cards,
            "drawn_cards": drawn_cards,
            "player_cards": player_cards,
            "community_cards": obs["community_cards"],
            "acting_agent": 0 if self.i_am_small_blind else 1,
            "small_blind_player": 0,
            "big_blind_player": 1,
            "min_raise": obs["min_raise"],
            "last_street_bet": self.last_street_bet,
            "terminated": False,
            "winner": None,  # 0 or 1 for a win, -1 for tie
        }
        return state

    def compute_information_set(self, obs):
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
        player = 0 if self.i_am_small_blind else 1# obs["acting_agent"]
        my_cards_sorted = "-".join(map(str, sorted(obs["my_cards"])))
        community_cards_sorted = "-".join(map(str, flop_cards_sorted + [turn_card, river_card]))
        my_bet_binned = int(math.log2(obs["my_bet"])) if obs["my_bet"] > 0 else 0
        opp_bet_binned = int(math.log2(obs["opp_bet"])) if obs["opp_bet"] > 0 else 0
        min_raise_binned = int(math.log2(obs["min_raise"])) if obs["min_raise"] > 0 else 0
        max_raise_binned = int(math.log2(obs["max_raise"])) if obs["max_raise"] > 0 else 0
        valid_actions = "".join(map(str, obs["valid_actions"]))

        # TODO have the exact cards discarded and drawn later
        i_discarded = obs["my_discarded_card"] != -1
        opp_discarded = obs["opp_discarded_card"] != -1

        return f"{player}_{my_cards_sorted}_{community_cards_sorted}_{my_bet_binned}_{opp_bet_binned}_{min_raise_binned}_{max_raise_binned}_{valid_actions}_{i_discarded}_{opp_discarded}"
    
    def compute_information_set_reduced(self, obs):
        flop_cards_sorted = sorted(obs["community_cards"][:3])
        turn_card = obs["community_cards"][3]
        river_card = obs["community_cards"][4]
        player = obs["acting_agent"]
        my_cards_sorted = "-".join(map(str, sorted(obs["my_cards"])))
        community_cards_sorted = "-".join(map(str, flop_cards_sorted + [turn_card, river_card]))
        my_bet_binned = int(math.log2(obs["my_bet"])) if obs["my_bet"] > 0 else 0
        opp_bet_binned = int(math.log2(obs["opp_bet"])) if obs["opp_bet"] > 0 else 0
        min_raise_binned = int(math.log2(obs["min_raise"])) if obs["min_raise"] > 0 else 0
        max_raise_binned = int(math.log2(obs["max_raise"])) if obs["max_raise"] > 0 else 0
        valid_actions = "".join(map(str, obs["valid_actions"]))

        i_discarded = obs["my_discarded_card"] != -1
        opp_discarded = obs["opp_discarded_card"] != -1

        my_card_numbers_sorted = ",".join(sorted(map(lambda card: str(card % 9), obs["my_cards"])))
        is_suited = (obs["my_cards"][0] // 9) == (obs["my_cards"][1] // 9)
        flop_card_numbers_sorted = sorted(map(lambda card: str(card % 9) if card != -1 else str(-1), flop_cards_sorted))
        turn_card_number = str(turn_card % 9) if turn_card != -1 else str(-1)
        river_card_number = str(river_card % 9) if river_card != -1 else str(-1)
        community_card_numbers_sorted = ",".join(flop_card_numbers_sorted + [turn_card_number, river_card_number])
        suits_map = {}
        for card in obs["my_cards"] + flop_cards_sorted + [turn_card, river_card]:
            if card == -1:
                continue
            if card // 9 not in suits_map:
                suits_map[card // 9] = 0
            suits_map[card // 9] += 1

        is_four_flush = max(suits_map.values()) >= 4
        is_five_flush = max(suits_map.values()) >= 5

        flop_card_numbers = ",".join(flop_card_numbers_sorted)

        return f"{player}_{my_card_numbers_sorted}_{is_suited}_{is_four_flush}_{is_five_flush}_{flop_card_numbers}_{valid_actions}"

    
def action_str_to_action_tuple(action_str, observation):
    if action_str == "FOLD":
        return (ActionType.FOLD.value, 0, -1)
    elif action_str == "CHECK":
        return (ActionType.CHECK.value, 0, -1)
    elif action_str == "CALL":
        return (ActionType.CALL.value, 0, -1)
    elif action_str == "DISCARD_0":
        return (ActionType.DISCARD.value, 0, 0)
    elif action_str == "DISCARD_1":
        return (ActionType.DISCARD.value, 0, 1)
    elif action_str == "RAISE_MIN":
        min_raise = min(observation["min_raise"], observation["max_raise"])
        return (ActionType.RAISE.value, min_raise, -1)
    elif action_str == "RAISE_MAX":
        return (ActionType.RAISE.value, observation["max_raise"], -1)
    elif action_str == "RAISE_POT":
        max_raise = observation["max_raise"]
        min_raise = min(observation["min_raise"], max_raise)
        pot = observation["my_bet"] + observation["opp_bet"]
        safe_bet = max(min_raise, min(max_raise, pot))
        return (ActionType.RAISE.value, safe_bet, -1)
    elif action_str == "RAISE_HALF_POT":
        max_raise = observation["max_raise"]
        min_raise = min(observation["min_raise"], max_raise)
        pot = observation["my_bet"] + observation["opp_bet"]
        half_pot = pot // 2
        safe_bet = max(min_raise, min(max_raise, half_pot))
        return (ActionType.RAISE.value, safe_bet, -1)
    else:
        raise f"wtf - invalid action_str {action_str}"



from agents.agent import Agent
from gym_env import PokerEnv
import random
from treys import Evaluator

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card

class ChallengeOne(Agent):
    def __name__(self):
        return "PlayerAgent"

    def __init__(self, stream: bool = False):
        super().__init__(stream)
        self.evaluator = Evaluator()

    def act(self, observation, reward, terminated, truncated, info):
        # Log new street starts with important info
        if observation["street"] == 0:  # Preflop
            self.logger.debug(f"Hole cards: {[int_to_card(c) for c in observation['my_cards']]}")
        elif observation["community_cards"]:  # New community cards revealed
            visible_cards = [c for c in observation["community_cards"] if c != -1]
            if visible_cards:
                street_names = ["Preflop", "Flop", "Turn", "River"]
                self.logger.debug(f"{street_names[observation['street']]}: {[int_to_card(c) for c in visible_cards]}")

        my_cards = [int(card) for card in observation["my_cards"]]
        community_cards = [card for card in observation["community_cards"] if card != -1]
        opp_discarded_card = [observation["opp_discarded_card"]] if observation["opp_discarded_card"] != -1 else []
        opp_drawn_card = [observation["opp_drawn_card"]] if observation["opp_drawn_card"] != -1 else []

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

        # Run Monte Carlo simulation
        num_simulations = 2000
        wins = sum(
            evaluate_hand((my_cards, opp_drawn_card + drawn_cards[: 2 - len(opp_drawn_card)], community_cards + drawn_cards[2 - len(opp_drawn_card) :]))
            for _ in range(num_simulations)
            if (drawn_cards := random.sample(non_shown_cards, 7 - len(community_cards) - len(opp_drawn_card)))
        )
        equity = wins / num_simulations

        # Calculate pot odds
        continue_cost = observation["opp_bet"] - observation["my_bet"]
        pot_size = observation["my_bet"] + observation["opp_bet"]
        pot_odds = continue_cost / (continue_cost + pot_size) if continue_cost > 0 else 0

        self.logger.debug(f"Equity: {equity:.2f}, Pot odds: {pot_odds:.2f}")

        # Decision making
        raise_amount = 0
        card_to_discard = -1

        # Only log very significant decisions at INFO level
        if equity > 0.8 and observation["valid_actions"][action_types.RAISE.value]:
            raise_amount = min(int(pot_size * 0.75), observation["max_raise"])
            raise_amount = max(raise_amount, observation["min_raise"])
            action_type = action_types.RAISE.value
            if raise_amount > 20:  # Only log large raises
                self.logger.info(f"Large raise to {raise_amount} with equity {equity:.2f}")

        if observation["valid_actions"][action_types.DISCARD.value] == 1:
            # Run Monte Carlo simulation
            num_simulations = 1000
            kept_card_idx = 0 
            kept_card = my_cards[kept_card_idx]
            wins = sum(
                evaluate_hand(([kept_card] + drawn_cards[0:1], opp_drawn_card + drawn_cards[1: 3 - len(opp_drawn_card)], community_cards + drawn_cards[3 - len(opp_drawn_card) :]))
                for _ in range(num_simulations)
                if (drawn_cards := random.sample(non_shown_cards, 8 - len(community_cards) - len(opp_drawn_card)))
            )
            equity_if_discard_0 = wins / num_simulations

            
            kept_card_idx = 1
            kept_card = my_cards[kept_card_idx]
            wins = sum(
                evaluate_hand(([kept_card] + drawn_cards[0:1], opp_drawn_card + drawn_cards[1: 3 - len(opp_drawn_card)], community_cards + drawn_cards[3 - len(opp_drawn_card) :]))
                for _ in range(num_simulations)
                if (drawn_cards := random.sample(non_shown_cards, 8 - len(community_cards) - len(opp_drawn_card)))
            )
            equity_if_discard_1 = wins / num_simulations

            
            if max(equity_if_discard_0, equity_if_discard_1) > equity + 0.2:
                print(f"Discarding because better equity by doing that! {max(equity_if_discard_0, equity_if_discard_1) - equity}")

                action_type = action_types.DISCARD.value
                if equity_if_discard_0 > equity_if_discard_1:
                    card_to_discard = 1
                else:
                    card_to_discard = 0
            

        
        if equity >= pot_odds and observation["valid_actions"][action_types.CALL.value]:
            action_type = action_types.CALL.value
        elif observation["valid_actions"][action_types.CHECK.value]:
            action_type = action_types.CHECK.value
        elif observation["valid_actions"][action_types.DISCARD.value]:
            action_type = action_types.DISCARD.value
            card_to_discard = random.randint(0, 1)
            self.logger.debug(f"Discarding card {card_to_discard}: {int_to_card(my_cards[card_to_discard])}")
        else:
            action_type = action_types.FOLD.value
            if observation["opp_bet"] > 20:  # Only log significant folds
                self.logger.info(f"Folding to large bet of {observation['opp_bet']}")

        return action_type, raise_amount, card_to_discard

    def observe(self, observation, reward, terminated, truncated, info):
        if terminated and abs(reward) > 20:  # Only log significant hand results
            self.logger.info(f"Significant hand completed with reward: {reward}")

import pickle
import random

class AbstractGame:
    def get_initial_state(self):
        """Return the initial state of the game."""
        raise NotImplementedError

    def is_terminal(self, state):
        """Return True if state is terminal."""
        raise NotImplementedError

    def get_utility(self, state):
        """
        Return a dictionary mapping player IDs to their utility
        at the terminal state.
        """
        raise NotImplementedError

    def is_chance_node(self, state):
        """Return True if the state is a chance node (e.g. a random event)."""
        raise NotImplementedError

    def sample_chance_action(self, state):
        """For chance nodes, sample and return an action (or outcome)."""
        raise NotImplementedError

    def get_current_player(self, state):
        """Return the player (or chance) whose turn it is to act."""
        raise NotImplementedError

    def get_information_set(self, state, player):
        """
        Return a key (e.g. a string) that uniquely represents the player’s information set
        in this state.
        """
        raise NotImplementedError

    def get_actions(self, state):
        """
        Return a list of actions available at the given state.
        This allows the set of actions to vary with state.
        """
        raise NotImplementedError

    def apply_action(self, state, action):
        """Return the new state after the given action is applied."""
        raise NotImplementedError

    def get_players(self):
        """Return a list of players in the game."""
        raise NotImplementedError


class MCCFRTrainer:
    def __init__(self, game: AbstractGame):
        self.game = game
        # We now maintain regret and strategy sums per information set,
        # where each info set’s dictionary keys come from the available actions at that node.
        self.regretSum = {}   # Maps infoSet -> {action: cumulative regret}
        self.strategySum = {} # Maps infoSet -> {action: cumulative strategy probability}

    def reset(self):
        self.regretSum = {}
        self.strategySum = {}

    def get_strategy(self, infoSet, available_actions, realization_weight):
        """
        Compute current strategy at the information set using regret-matching,
        and update the cumulative strategy sum.
        """
        # Initialize regrets for this infoSet if unseen.
        if infoSet not in self.regretSum:
            self.regretSum[infoSet] = {a: 0.0 for a in available_actions}
        regrets = self.regretSum[infoSet]
        normalizing_sum = sum(max(regrets[a], 0) for a in available_actions)
        if normalizing_sum > 0:
            strategy = {a: max(regrets[a], 0) / normalizing_sum for a in available_actions}
        else:
            strategy = {a: 1.0 / len(available_actions) for a in available_actions}

        # Update cumulative strategy sum.
        if infoSet not in self.strategySum:
            self.strategySum[infoSet] = {a: 0.0 for a in available_actions}
        else:
            # Make sure all available actions are present.
            for a in available_actions:
                if a not in self.strategySum[infoSet]:
                    self.strategySum[infoSet][a] = 0.0

        for a in available_actions:
            self.strategySum[infoSet][a] += realization_weight * strategy[a]
        return strategy

    def sample_action(self, strategy):
        """Sample an action based on the provided strategy (a dict mapping actions to probabilities)."""
        r = random.random()
        cumulative_probability = 0.0
        for a, prob in strategy.items():
            cumulative_probability += prob
            if r < cumulative_probability:
                return a
        return list(strategy.keys())[-1]  # Fallback in case of rounding issues

    def cfr(self, state, reach_probs, sample_probs):
        """
        Recursively perform outcome sampling MCCFR on the state.
          - `reach_probs`: dict mapping player to probability of reaching state under current strategy.
          - `sample_probs`: dict mapping player to probability under sampling.
        Returns a dictionary of counterfactual values for each player.
        """
        if self.game.is_terminal(state):
            ut =  self.game.get_utility(state)
            # print(ut)
            return ut
        
        if self.game.is_chance_node(state):
            # Sample a chance action and continue.
            action = self.game.sample_chance_action(state)
            next_state = self.game.apply_action(state, action)
            return self.cfr(next_state, reach_probs, sample_probs)

        current_player = self.game.get_current_player(state)
        infoSet = self.game.get_information_set(state, current_player)
        available_actions = self.game.get_actions(state)
        strategy = self.get_strategy(infoSet, available_actions, reach_probs[current_player])
        
        # Outcome sampling: sample one action according to the current strategy.
        sampled_action = self.sample_action(strategy)
        new_state = self.game.apply_action(state, sampled_action)

        # Update sampling probability for the current player.
        new_sample_probs = sample_probs.copy()
        new_sample_probs[current_player] *= strategy[sampled_action]

        # Recursively compute counterfactual utilities.
        utilities = self.cfr(new_state, reach_probs, new_sample_probs)
        util_current = utilities[current_player]

        # Update regrets only for the actions available at this state.
        for a in available_actions:
            if a == sampled_action:
                # Importance sampling correction: divide by probability of sampling a.
                action_util = util_current / strategy[a]
            else:
                action_util = 0
            regret = action_util - util_current
            # Initialize regretSum for infoSet if necessary.
            if infoSet not in self.regretSum:
                self.regretSum[infoSet] = {act: 0.0 for act in available_actions}
            self.regretSum[infoSet][a] += (1.0 / sample_probs[current_player]) * regret

        return utilities

    def train(self, iterations: int, save_strat_sum_every = 10_000_000, custom_initial_state = None):
        """
        Run MCCFR for a specified number of iterations.
        Returns the average strategy for each information set.
        """
        players = self.game.get_players()
        for i in range(iterations):
            reach_probs = {p: 1.0 for p in players}
            sample_probs = {p: 1.0 for p in players}
            initial_state = self.game.get_initial_state() if custom_initial_state is None else custom_initial_state
            self.cfr(initial_state, reach_probs, sample_probs)

            if i % 10000 == 0:
                print(f"Iteration {i} - Number of infosets recorded: {len(self.strategySum)}")

            if i % save_strat_sum_every == 0:
                with open(f"strat_sum_{i}.pkl", "wb") as f:
                    pickle.dump(self.strategySum, f)

        # Compute average strategy from cumulative strategy sums.
        average_strategy = {}
        for infoSet, strat_sum in self.strategySum.items():
            total = sum(strat_sum.values())
            if total > 0:
                average_strategy[infoSet] = {a: strat_sum[a] / total for a in strat_sum}
            else:
                # In case no strategy was accumulated, default to uniform over the actions seen.
                n = len(strat_sum)
                average_strategy[infoSet] = {a: 1.0 / n for a in strat_sum}
        return average_strategy

    def train_strategy_sum(self, iterations: int, save_strat_sum_every = 10_000_000, custom_initial_state = None):
        """
        Run MCCFR for a specified number of iterations.
        Returns the average strategy for each information set.
        """
        players = self.game.get_players()
        for i in range(iterations):
            reach_probs = {p: 1.0 for p in players}
            sample_probs = {p: 1.0 for p in players}
            initial_state = self.game.get_initial_state() if custom_initial_state is None else custom_initial_state
            self.cfr(initial_state, reach_probs, sample_probs)

            if i % 10000 == 0 and i > 0:
                print(f"Iteration {i} - Number of infosets recorded: {len(self.strategySum)}")

        return self.regretSum, self.strategySum
        # Compute average strategy from cumulative strategy sums.
        average_strategy = {}
        for infoSet, strat_sum in self.strategySum.items():
            total = sum(strat_sum.values())
            if total > 0:
                average_strategy[infoSet] = {a: strat_sum[a] / total for a in strat_sum}
            else:
                # In case no strategy was accumulated, default to uniform over the actions seen.
                n = len(strat_sum)
                average_strategy[infoSet] = {a: 1.0 / n for a in strat_sum}
        return average_strategy

# A helper function that runs a training batch in a separate process.
def train_batch(args):
    game, iterations, custom_initial_state = args
    trainer = MCCFRTrainer(game)
    regretSum, strategySum = trainer.train_strategy_sum(iterations, custom_initial_state)
    return regretSum, strategySum

# A helper to merge dictionaries of the form {infoSet: {action: value}}
def merge_updates(updates):
    merged_regretSum = {}
    merged_strategySum = {}
    for regretSum, strategySum in updates:
        for infoSet, action_dict in regretSum.items():
            if infoSet not in merged_regretSum:
                merged_regretSum[infoSet] = {}
            for a, val in action_dict.items():
                merged_regretSum[infoSet][a] = merged_regretSum[infoSet].get(a, 0.0) + val
        for infoSet, action_dict in strategySum.items():
            if infoSet not in merged_strategySum:
                merged_strategySum[infoSet] = {}
            for a, val in action_dict.items():
                merged_strategySum[infoSet][a] = merged_strategySum[infoSet].get(a, 0.0) + val
    return merged_regretSum, merged_strategySum


import enum
import copy
import math
import torch
import numpy as np
from treys import Card, Evaluator

# --- WrappedEval ---
class WrappedEval(Evaluator):
    def __init__(self):
        super().__init__()

    def evaluate(self, hand: list[int], board: list[int]) -> int:
        """
        Evaluates a hand with a twist: it also computes an alternate score
        where aces are treated as tens. Returns the lower score.
        """
        def ace_to_ten(treys_card: int):
            s = Card.int_to_str(treys_card)
            alt = s.replace("A", "T")  # Convert Ace to Ten
            return Card.new(alt)

        alt_hand = list(map(ace_to_ten, hand))
        alt_board = list(map(ace_to_ten, board))
        reg_score = super().evaluate(hand, board)
        alt_score = super().evaluate(alt_hand, alt_board)
        if alt_score < reg_score:
            return alt_score
        return reg_score

# --- ActionType enum ---
class ActionType(enum.Enum):
    FOLD = 0
    RAISE = 1
    CHECK = 2
    CALL = 3
    DISCARD = 4
    INVALID = 5

# --- PokerGame: the independent poker engine ---
class PokerGame(AbstractGame):
    def __init__(self, small_blind_amount: int = 1, max_player_bet: int = 100):
        self.small_blind_amount = small_blind_amount
        self.big_blind_amount = small_blind_amount * 2
        self.max_player_bet = max_player_bet
        self.RANKS = "23456789A"
        self.SUITS = "dhs"  # diamonds, hearts, spades
        self.evaluator = WrappedEval()

    # ----- Helper: Convert an integer card to a treys card -----
    def int_to_card(self, card_int: int) -> int:
        rank = self.RANKS[card_int % len(self.RANKS)]
        suit = self.SUITS[card_int // len(self.RANKS)]
        return Card.new(rank + suit)

    # ----- Game initialization -----
    def get_initial_state(self):
        seed = torch.randint(0, 1_000_000_000, (1,)).item()
        rng = np.random.RandomState(seed)
        # Create deck: our game uses 27 cards (integers 0..26)
        deck = list(np.arange(27))
        rng.shuffle(deck)
        # Set blinds (by default player 0 is small blind and player 1 is big blind)
        small_blind_player = 0
        big_blind_player = 1
        # Deal two cards to each player
        player_cards = [[deck.pop(0) for _ in range(2)] for _ in range(2)]
        # Deal five community cards
        community_cards = [deck.pop(0) for _ in range(5)]
        # Set initial bets (blinds)
        bets = [0, 0]
        bets[small_blind_player] = self.small_blind_amount
        bets[big_blind_player] = self.big_blind_amount
        state = {
            "seed": seed,
            "deck": deck,  # remaining deck
            "street": 0,
            "bets": bets,
            "discarded_cards": [-1, -1],
            "drawn_cards": [-1, -1],
            "player_cards": player_cards,
            "community_cards": community_cards,
            "acting_agent": small_blind_player,
            "small_blind_player": small_blind_player,
            "big_blind_player": big_blind_player,
            "min_raise": self.big_blind_amount,
            "last_street_bet": 0,
            "terminated": False,
            "winner": None,  # 0 or 1 for a win, -1 for tie
        }
        return state

    # ----- Terminal state & utilities -----
    def is_terminal(self, state):
        return state.get("terminated", False)

    def get_utility(self, state):
        if not self.is_terminal(state):
            raise ValueError("Game is not terminated yet.")
        pot = min(state["bets"])
        if state["winner"] == 0:
            return {0: pot, 1: -pot}
        elif state["winner"] == 1:
            return {0: -pot, 1: pot}
        else:  # tie
            return {0: 0, 1: 0}

    # ----- Chance nodes (none in this game) -----
    def is_chance_node(self, state):
        return False

    def sample_chance_action(self, state):
        raise NotImplementedError("This game has no chance nodes.")

    # ----- Current player & information sets -----
    def get_current_player(self, state):
        return state["acting_agent"]

    def _get_single_player_obs(self, state, player):
        # Determine how many community cards are revealed.
        if state["street"] == 0:
            num_cards_to_reveal = 0
        else:
            num_cards_to_reveal = state["street"] + 2  # as in gym_env
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
        # Ensure all-in situations are handled.
        if obs["min_raise"] > obs["max_raise"]:
            obs["min_raise"] = obs["max_raise"]
        return obs

    def get_information_set(self, state, player):
        # For simplicity, we use the string representation of the player's observation.
        obs = self._get_single_player_obs(state, player)
        return PokerGame.compute_information_set(obs)

    def get_actions(self, state):
        acting = state["acting_agent"]
        valid = self._get_valid_actions(state, acting)
        actions = []
        # FOLD
        if valid[ActionType.FOLD.value]:
            actions.append("FOLD")
        # CHECK
        if valid[ActionType.CHECK.value]:
            actions.append("CHECK")
        # CALL
        if valid[ActionType.CALL.value]:
            actions.append("CALL")
        # DISCARD (two options: discard card 0 or 1)
        if valid[ActionType.DISCARD.value]:
            actions.append("DISCARD_0")
            actions.append("DISCARD_1")
        # RAISE: include options for min, max, and if possible intermediate values.
        if valid[ActionType.RAISE.value]:
            obs = self._get_single_player_obs(state, acting)
            min_raise = obs["min_raise"]
            max_raise = obs["max_raise"]
            pot = state["bets"][0] + state["bets"][1]
            actions.append("RAISE_MIN")
            actions.append("RAISE_MAX")
            # if min_raise < pot < max_raise:
            actions.append("RAISE_POT")
            # if min_raise < pot // 2 < max_raise:
            actions.append("RAISE_HALF_POT")
        return actions
    
    def action_str_to_action_tuple(self, state, action_str):
        if action_str == "FOLD":
            return (ActionType.FOLD.value, 0, -1)
        elif action_str == "CHECK":
            return (ActionType.CHECK.value, 0, -1)
        elif action_str == "CALL":
            return (ActionType.CALL.value, 0, -1)
        elif action_str == "DISCARD_0":
            return (ActionType.DISCARD.value, 0, 0)
        elif action_str == "DISCARD_1":
            return (ActionType.DISCARD.value, 0, 1)
        elif action_str == "RAISE_MIN":
            min_raise = min(state["min_raise"], self.max_player_bet - max(state["bets"]))
            return (ActionType.RAISE.value, min_raise, -1)
        elif action_str == "RAISE_MAX":
            return (ActionType.RAISE.value, self.max_player_bet - max(state["bets"]), -1)
        elif action_str == "RAISE_POT":
            max_raise = self.max_player_bet - max(state["bets"])
            min_raise = min(state["min_raise"], max_raise)
            pot = sum(state["bets"])
            safe_bet = max(min_raise, min(max_raise, pot))
            return (ActionType.RAISE.value, safe_bet, -1)
        elif action_str == "RAISE_HALF_POT":
            max_raise = self.max_player_bet - max(state["bets"])
            min_raise = min(state["min_raise"], max_raise)
            pot = sum(state["bets"])
            half_pot = pot // 2
            safe_bet = max(min_raise, min(max_raise, half_pot))
            return (ActionType.RAISE.value, safe_bet, -1)
        else:
            raise f"wtf - invalid action_str {action_str}"

    def get_players(self):
        return [0, 1]

    # ----- Helper: Valid actions -----
    def _get_valid_actions(self, state, player):
        # The order of actions is: FOLD, RAISE, CHECK, CALL, DISCARD.
        valid = [1, 1, 1, 1, 1]
        opponent = 1 - player
        # Cannot check if behind
        if state["bets"][player] < state["bets"][opponent]:
            valid[ActionType.CHECK.value] = 0
        # Cannot call if already equal
        if state["bets"][player] == state["bets"][opponent]:
            valid[ActionType.CALL.value] = 0
        # Can discard only if not already discarded and only in early streets
        if state["discarded_cards"][player] != -1:
            valid[ActionType.DISCARD.value] = 0
        if state["street"] > 1:
            valid[ActionType.DISCARD.value] = 0
        # Cannot raise if a player is all in
        if max(state["bets"]) == self.max_player_bet:
            valid[ActionType.RAISE.value] = 0
        return valid

    # ----- Core game logic: apply an action -----
    def apply_action(self, state, action_str):
        action = self.action_str_to_action_tuple(state, action_str)
        # Make a deep copy so that previous states remain unchanged.
        new_state = copy.deepcopy(state)
        if new_state["terminated"]:
            raise ValueError("Cannot apply action: game is already terminated.")
        a_type, raise_amount, card_to_discard = action
        valid = self._get_valid_actions(new_state, new_state["acting_agent"])
        # If an invalid action is attempted, treat it as a fold.
        if not valid[a_type]:
            a_type = ActionType.INVALID.value
        # For a raise, check the raise amount.
        if a_type == ActionType.RAISE.value:
            if not (new_state["min_raise"] <= raise_amount <= (self.max_player_bet - max(new_state["bets"]))):
                a_type = ActionType.INVALID.value

        winner = None
        new_street = False
        current = new_state["acting_agent"]
        opponent = 1 - current

        if a_type in (ActionType.FOLD.value, ActionType.INVALID.value):
            # Treat fold or invalid action as a fold.
            winner = opponent
            new_state["terminated"] = True
            new_state["winner"] = winner
        elif a_type == ActionType.CALL.value:
            new_state["bets"][current] = new_state["bets"][opponent]
            # On the first street, the small blind calling the big blind does not advance the street.
            if not (new_state["street"] == 0 and current == new_state["small_blind_player"] and new_state["bets"][current] == self.big_blind_amount):
                new_street = True
        elif a_type == ActionType.CHECK.value:
            if current == new_state["big_blind_player"]:
                new_street = True  # Big blind checking advances the street.
        elif a_type == ActionType.RAISE.value:
            new_state["bets"][current] = new_state["bets"][opponent] + raise_amount
            raise_so_far = new_state["bets"][opponent] - new_state["last_street_bet"]
            max_raise = self.max_player_bet - max(new_state["bets"])
            min_raise_no_limit = raise_so_far + raise_amount
            new_state["min_raise"] = min(min_raise_no_limit, max_raise)
        elif a_type == ActionType.DISCARD.value:
            if card_to_discard != -1:
                new_state["discarded_cards"][current] = new_state["player_cards"][current][card_to_discard]
                if new_state["deck"]:
                    drawn = new_state["deck"].pop(0)
                else:
                    drawn = -1
                new_state["drawn_cards"][current] = drawn
                new_state["player_cards"][current][card_to_discard] = drawn

        # Advance to next street if needed.
        if new_street:
            new_state["street"] += 1
            new_state["min_raise"] = self.big_blind_amount
            new_state["last_street_bet"] = new_state["bets"][0]  # bets should be equal at this point.
            new_state["acting_agent"] = new_state["small_blind_player"]
            if new_state["street"] > 3:
                # Showdown: determine winner.
                winner = self._get_winner(new_state)
                new_state["terminated"] = True
                new_state["winner"] = winner
        elif a_type != ActionType.DISCARD.value:
            new_state["acting_agent"] = opponent

        # Recalculate min_raise to enforce all-in limits.
        new_state["min_raise"] = min(new_state["min_raise"], self.max_player_bet - max(new_state["bets"]))
        return new_state

    # ----- Determine the winner at showdown -----
    def _get_winner(self, state):
        board = [self.int_to_card(c) for c in state["community_cards"] if c != -1]
        while len(board) < 5:
                board.append(self.int_to_card(state["deck"].pop(0)))
        p0_cards = [self.int_to_card(c) for c in state["player_cards"][0] if c != -1]
        p1_cards = [self.int_to_card(c) for c in state["player_cards"][1] if c != -1]

        try:
            score0 = self.evaluator.evaluate(p0_cards, board)
            score1 = self.evaluator.evaluate(p1_cards, board)
            if score0 == score1:
                return -1  # Tie
            elif score1 < score0:
                return 1
            else:
                return 0
        except Exception as e:
            print(state["player_cards"][0], state["player_cards"][1], state["community_cards"], state["acting_agent"])
            raise e
        
    
    def compute_information_set(obs):
        flop_cards_sorted = sorted(obs["community_cards"][:3])
        turn_card = obs["community_cards"][3]
        river_card = obs["community_cards"][4]
        suits_map = {}
        for card in obs["my_cards"] + flop_cards_sorted + [turn_card, river_card]:
            if card == -1:
                continue
            if card // 9 not in suits_map:
                suits_map[card // 9] = 0
            suits_map[card // 9] += 1

        is_four_flush = max(suits_map.values()) >= 4
        is_five_flush = max(suits_map.values()) >= 5

        convert_card_to_0_to_9_number = lambda card: 1 + ((card+1) % 9 if card != -1 else -1)

        my_card_numbers_sorted_0_to_9 = sorted(map(convert_card_to_0_to_9_number, obs["my_cards"]))
        community_card_numbers_sorted_0_to_9 = sorted(map(convert_card_to_0_to_9_number, obs["community_cards"]))
        valid_actions = "".join(map(str, obs["valid_actions"]))
        VALID_ACTIONS_MAP = {
            "11101": 0,
            "10010": 1,
            "11011": 2,
            "10101": 3,
            "11010": 4,
            "11100": 5,
            "10100": 6,
            "10011": 7,
        }

        continuation_cost = obs["opp_bet"] - obs["my_bet"]
        pot = obs["opp_bet"] + obs["my_bet"]
        pot_odds = continuation_cost / (continuation_cost + pot)

        player = obs["acting_agent"]
        my_hand_numbers_int = tuple_to_int_2(my_card_numbers_sorted_0_to_9)
        are_my_two_cards_suited_0_to_1 = 1 if (obs["my_cards"][0] // 9) == (obs["my_cards"][1] // 9) else 0
        flush_number = 0 if not is_four_flush else (1 if not is_five_flush else 2)
        community_card_numbers_int = tuple_to_int_5(community_card_numbers_sorted_0_to_9)
        valid_actions_number = VALID_ACTIONS_MAP[valid_actions]
        binned_pot_odds = int(pot_odds * 3)

        fields = (player, my_hand_numbers_int, are_my_two_cards_suited_0_to_1, flush_number, community_card_numbers_int, valid_actions_number, binned_pot_odds)
        radices = (2    , 55                 , 2                             , 3           , 2002                      , 8                   , 3              )
        info_set_index = encode_fields(fields, radices)

        return info_set_index

from functools import reduce

def encode_fields(values, radices):
    """
    Encode a list of values into a single integer using a mixed-radix system.
    
    :param values: A list/tuple of integers, where each integer is in 0 to N-1.
    :param radices: A list/tuple of the number of possibilities for each value.
    :return: An integer encoding the combination.
    """
    # Assuming values and radices are ordered from most significant to least.
    return reduce(lambda acc, pair: acc * pair[1] + pair[0], zip(values, radices), 0)

def decode_fields(index, radices):
    """
    Decode an integer back into the list of values.
    
    :param index: The encoded integer.
    :param radices: A list/tuple of the number of possibilities for each value.
    :return: A tuple of integers corresponding to the original values.
    """
    values = []
    for radix in reversed(radices):
        values.append(index % radix)
        index //= radix
    return tuple(reversed(values))

def tuple_to_int_5(t):
    """
    Maps a 5-tuple of integers (0-9) (order doesn't matter, so it's sorted)
    with repetition allowed to a unique integer in 0..2001.
    
    Steps:
      1. Sort the tuple in non-decreasing order.
      2. Transform each element: y[i] = sorted_tuple[i] + i, which yields a strictly increasing sequence.
      3. Rank the combination y among all 5-combinations of numbers from 0 to 13.
    """
    # Ensure a canonical order.
    t_sorted = sorted(t)
    # Transform to a strictly increasing tuple.
    y = [t_sorted[i] + i for i in range(5)]
    
    # n is now 10 (possible original values) + 5 - 1 = 14, so valid numbers are 0..13.
    n = 14  
    k = 5
    rank = 0
    prev = 0
    # For each position, count how many combinations come before the given number.
    for i in range(k):
        for j in range(prev, y[i]):
            rank += math.comb(n - j - 1, k - i - 1)
        prev = y[i] + 1
    return rank

def int_to_tuple_5(rank):
    """
    Inverse of tuple_to_int.
    
    Given an integer in 0..2001, returns the corresponding 5-tuple
    (sorted in non-decreasing order) of integers in 0-9 (allowing repetitions).
    
    Steps:
      1. Unrank the number into a strictly increasing 5-tuple y in {0,...,13}.
      2. Reverse the transformation: x[i] = y[i] - i.
    """
    n = 14  # numbers range 0 to 13 now.
    k = 5
    y = []
    x_val = 0
    for i in range(k):
        while True:
            count = math.comb(n - x_val - 1, k - i - 1)
            if rank < count:
                y.append(x_val)
                x_val += 1
                break
            else:
                rank -= count
                x_val += 1
    # Reverse the transformation to obtain the original tuple.
    original_tuple = tuple(y[i] - i for i in range(k))
    return original_tuple

def tuple_to_int_2(t):
    """
    Maps a 2-tuple of integers (0-9) (order doesn't matter, so it's sorted)
    with repetition allowed to a unique integer in 0..54.

    Steps:
      1. Sort the tuple in non-decreasing order.
      2. Transform: y[i] = sorted_tuple[i] + i to get a strictly increasing tuple.
      3. Rank the combination y among all 2-combinations of numbers from 0 to 10.
    """
    # Step 1: sort the tuple
    t_sorted = sorted(t)
    # Step 2: transform to strictly increasing tuple
    y = [t_sorted[i] + i for i in range(2)]
    
    # There are n = 10 + 2 - 1 = 11 numbers (0 to 10) to choose from
    n = 11  
    k = 2
    rank = 0
    prev = 0
    # Step 3: Compute the lexicographic rank
    for i in range(k):
        for j in range(prev, y[i]):
            rank += math.comb(n - j - 1, k - i - 1)
        prev = y[i] + 1
    return rank

def int_to_tuple_2(rank):
    """
    Inverse of tuple_to_int_2.
    
    Given an integer in 0..54, returns the corresponding 2-tuple
    (sorted in non-decreasing order) of integers in 0-9 (allowing repetitions).
    
    Steps:
      1. Unrank the number into a strictly increasing 2-tuple y in {0,...,10}.
      2. Reverse the transformation: x[i] = y[i] - i.
    """
    n = 11  # Numbers are from 0 to 10.
    k = 2
    y = []
    x_val = 0
    # Unranking: determine each y[i]
    for i in range(k):
        while True:
            count = math.comb(n - x_val - 1, k - i - 1)
            if rank < count:
                y.append(x_val)
                x_val += 1
                break
            else:
                rank -= count
                x_val += 1
    # Reverse the transformation
    original_tuple = tuple(y[i] - i for i in range(k))
    return original_tuple
