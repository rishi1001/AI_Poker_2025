import pickle
import time

import numpy as np
import torch
from agents.agent import Agent
from cfr import mccfr, poker_game
from gym_env import PokerEnv
import random
import math

import gym_env

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
        self.about_to_begin = True
        self.i_am_small_blind = True
        self.last_street_bet = 0
        self.last_street = 0
        self.cumulative_profit = 0

        self.chal_one = ChallengeOne()

        file_name = "strat_tables/merged_strategy_100000000.pkl"
        
        print(f"Loading avg_strategy from {file_name}")
        # Load avg_strategy from the pickle file
        with open(file_name, "rb") as f:
            self.avg_strategy = pickle.load(f)
        print(f"Loaded avg_strategy from {file_name}")

    def act(self, observation, reward, terminated, truncated, info):
        if self.cumulative_profit > 0.5 + (1000-self.hand_number) * 1.5:
            return (poker_game.ActionType.FOLD.value, 0, -1)
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
        game = poker_game.PokerGame()
        trainer = mccfr.MCCFRTrainer(game)
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
        return (poker_game.ActionType.FOLD.value, 0, -1)
    elif action_str == "CHECK":
        return (poker_game.ActionType.CHECK.value, 0, -1)
    elif action_str == "CALL":
        return (poker_game.ActionType.CALL.value, 0, -1)
    elif action_str == "DISCARD_0":
        return (poker_game.ActionType.DISCARD.value, 0, 0)
    elif action_str == "DISCARD_1":
        return (poker_game.ActionType.DISCARD.value, 0, 1)
    elif action_str == "RAISE_MIN":
        min_raise = min(observation["min_raise"], observation["max_raise"])
        return (poker_game.ActionType.RAISE.value, min_raise, -1)
    elif action_str == "RAISE_MAX":
        return (poker_game.ActionType.RAISE.value, observation["max_raise"], -1)
    elif action_str == "RAISE_POT":
        max_raise = observation["max_raise"]
        min_raise = min(observation["min_raise"], max_raise)
        pot = observation["my_bet"] + observation["opp_bet"]
        safe_bet = max(min_raise, min(max_raise, pot))
        return (poker_game.ActionType.RAISE.value, safe_bet, -1)
    elif action_str == "RAISE_HALF_POT":
        max_raise = observation["max_raise"]
        min_raise = min(observation["min_raise"], max_raise)
        pot = observation["my_bet"] + observation["opp_bet"]
        half_pot = pot // 2
        safe_bet = max(min_raise, min(max_raise, half_pot))
        return (poker_game.ActionType.RAISE.value, safe_bet, -1)
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
        num_simulations = 3000
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
        elif equity >= pot_odds and observation["valid_actions"][action_types.CALL.value]:
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
