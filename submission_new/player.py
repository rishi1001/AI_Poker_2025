from functools import reduce
import pickle

import numpy as np
from agents.agent import Agent
from gym_env import PokerEnv
import random
from treys import Evaluator, Card

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card

ARGMAX_SAMPLING = True

class PlayerAgentExperimental(Agent):
    def __name__(self):
        return "GigaReducedMCCFR"

    def __init__(self, stream: bool = False):
        super().__init__(stream)
        self.evaluator = Evaluator()

        self.chal_one = ChallengeOne()

        self.hand_number = 0
        self.cumulative_profit = 0

        file_name = "submission_new/merged_avg_strategy.pkl"
        
        print(f"Loading avg_strategy from {file_name}")
        # Load avg_strategy from the pickle file
        with open(file_name, "rb") as f:
            self.avg_strategy = pickle.load(f)
        print(f"Loaded avg_strategy from {file_name} - length: {len(self.avg_strategy)}")

    def act(self, observation, reward, terminated, truncated, info):
        if self.cumulative_profit > 0.5 + (1000-self.hand_number) * 1.5:
            return (action_types.FOLD.value, 0, -1)
        # compute info_set { equity_binned, flush_number, valid_actions_number, binned_pot_odds }

        info_set = self.compute_info_set(observation)
        info_set_integer = self.info_set_to_integer(info_set)

        # look up in avg_strategy
        if self.avg_strategy[info_set_integer] is not None and not all([x == 1/len(self.avg_strategy[info_set_integer]) for x in self.avg_strategy[info_set_integer]]):
            action_distribution = self.avg_strategy[info_set_integer]

            action_int = self.sample_action_from_distribution(action_distribution)

            action_tuple = self.action_int_to_action(action_int, observation)

            return self.safety_check(action_tuple, info_set, observation)
        # otherwise, log it and fallback to chal 1
        print(f"Info set not found in avg_strategy: {info_set}")

        return self.chal_one.act(observation, reward, terminated, truncated, info)

    def observe(self, observation, reward, terminated, truncated, info):
        if terminated: 

            self.hand_number += 1
            self.cumulative_profit += reward

            # if abs(reward) > 20: # Only log significant hand results
            #     self.logger.info(f"Significant hand completed with reward: {reward} - {list(map(lambda c: Card.int_to_pretty_str(int_to_card(c)), observation["my_cards"]))} - {list(map(lambda c: Card.int_to_pretty_str(int_to_card(c)), observation["community_cards"]))} - {info["player_1_cards"]}")


    def compute_info_set(self, observation):
        street = observation["street"]
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

        if street <= 1:
            # special binning
            thresholds = [0.3505, 0.377, 0.3945, 0.409, 0.427, 0.4395, 0.455, 0.4705, 0.483, 0.503, 0.5195, 0.531, 0.5425, 0.557, 0.5785]
            binned_equity = 15
            for i, threshold in enumerate(thresholds):
                if equity <= threshold:
                    binned_equity = i
                    break
        else:
            binned_equity = min(15, int(equity * 16))


        suits_map = {}
        for card in my_cards + community_cards + opp_discarded_card + opp_drawn_card:
            if card == -1:
                continue
            if card // 9 not in suits_map:
                suits_map[card // 9] = 0
            suits_map[card // 9] += 1
        max_count = max(suits_map.values(), default=0)
        is_four_flush = max_count >= 4
        is_five_flush = max_count >= 5
        flush_number = 0 if not is_four_flush else (1 if not is_five_flush else 2)

        valid_actions_str = "".join(map(str, observation["valid_actions"]))
        valid_actions_map = {
            "11101": 0,
            "10010": 1,
            "11011": 2,
            "10101": 3,
            "11010": 4,
            "11100": 5,
            "10100": 6,
            "10011": 7,
        }
        valid_actions_number = valid_actions_map.get(valid_actions_str, 0)
        

        continue_cost = observation["opp_bet"] - observation["my_bet"]
        pot_size = observation["my_bet"] + observation["opp_bet"]
        pot_odds = continue_cost / pot_size if continue_cost > 0 else 0
        binned_pot_odds = int(pot_odds * 8)


        return (street, binned_equity, valid_actions_number, binned_pot_odds)
    
    def info_set_to_integer(self, info_set):
        radices = [4, 16, 8, 8]
        return self.encode_fields(info_set, radices)
    def encode_fields(self, values, radices):
        """
        Encode a list of values into a single integer using a mixed-radix system.
        
        :param values: A list/tuple of integers, where each integer is in 0 to N-1.
        :param radices: A list/tuple of the number of possibilities for each value.
        :return: An integer encoding the combination.
        """
        # Assuming values and radices are ordered from most significant to least.
        return reduce(lambda acc, pair: acc * pair[1] + pair[0], zip(values, radices), 0)
    
    def sample_action_from_distribution(self, action_distribution):
        if not ARGMAX_SAMPLING:
            return np.random.choice(len(action_distribution), p=action_distribution)

        # # just take action with highest probability
        # # but first sum up all raise probs, from idx 5 to 11 included
        # raise_prob = 0
        # for i in range(5, 12):
        #     raise_prob += action_distribution[i]

        # if raise_prob > 0.5:
        #     # sample from the relative probabilities, first renormalize
        #     renormalized_probs = np.array(action_distribution[5:]) / raise_prob
        #     return np.random.choice(len(renormalized_probs), p=renormalized_probs) + 5
        return np.argmax(action_distribution)

    def action_int_to_action(self, action_int, observation):
        my_hand = observation["my_cards"]
        max_raise = observation["max_raise"]
        min_raise = min(observation["min_raise"], max_raise)
        my_bet = observation["my_bet"]
        opp_bet = observation["opp_bet"]
        if action_int == 0:
            return (action_types.FOLD, 0, -1)
        elif action_int == 1:
            return (action_types.CHECK, 0, -1)
        elif action_int == 2:
            return (action_types.CALL, 0, -1)
        elif action_int == 3 or action_int == 4: # Discard lower card
            lower_card_idx = 0 if my_hand[0] % 9 <= my_hand[1] % 9 else 1
            return (action_types.DISCARD, 0, lower_card_idx)
        elif action_int == 4: # Discard higher card
            higher_card_idx = 0 if my_hand[0] % 9 >= my_hand[1] % 9 else 1
            return (action_types.DISCARD, 0, higher_card_idx)
        elif action_int == 5: # Raise (min_raise)
            min_raise = min(min_raise, max_raise)
            return (action_types.RAISE, min_raise, -1)
        elif action_int == 6: # Raise (max_raise)
            return (action_types.RAISE, max_raise, -1)
        elif action_int == 7: # Raise (pot)
            pot = my_bet + opp_bet
            mult = pot
            safe_bet = max(min_raise, min(max_raise, mult))
            return (action_types.RAISE, safe_bet, -1)
        elif action_int == 8: # Raise (half pot)
            pot = my_bet + opp_bet
            safe_bet = max(min_raise, min(max_raise, pot))
            return (action_types.RAISE, safe_bet, -1)
        elif action_int == 9: # Raise uniform(1 * pot, 2 * pot)
            pot = my_bet + opp_bet
            mult = random.randrange(1 * pot, 2 * pot)
            safe_bet = max(min_raise, min(max_raise, mult))
            return (action_types.RAISE, safe_bet, -1)
        elif action_int == 10: # Raise uniform(2 * pot, 4 * pot)
            pot = my_bet + opp_bet
            mult = random.randrange(2 * pot, 4 * pot)
            safe_bet = max(min_raise, min(max_raise, mult))
            return (action_types.RAISE, safe_bet, -1)
        elif action_int == 11: # Raise uniform(4 * pot, max_raise)
            pot = my_bet + opp_bet
            mult = random.randrange(4 * pot, max(4 * pot + 1, max_raise))
            safe_bet = max(min_raise, min(max_raise, mult))
            return (action_types.RAISE, safe_bet, -1)
        else:
            raise ValueError(f"Invalid action integer: {action_int}")
        
    def safety_check(self, action, info_set, observation):
        street, binned_equity, flush_number, binned_pot_odds = info_set
        if binned_equity >= 15:
            if action[0].value != action_types.RAISE.value:
                if observation["valid_actions"][action_types.RAISE.value] == 1:
                    print(f"big binned equity and we were going to {action[0].name} - raising x*pot instead")
                    pot = observation["my_bet"] + observation["opp_bet"]
                    rand_bet = random.randrange(1, 7) * pot
                    return (action_types.RAISE, max(observation["min_raise"], min(observation["max_raise"], (observation["my_bet"]+observation["opp_bet"])*4)), -1)
                if observation["valid_actions"][action_types.CALL.value] == 1:
                    return (action_types.CALL, 0, -1)
                if observation["valid_actions"][action_types.CHECK.value] == 1:
                    return (action_types.CHECK, 0, -1)
                
        # if action[0].value == action_types.FOLD.value:
        #     if observation["valid_actions"][action_types.CHECK.value] == 1:
        #         print("YO")
        #         return (action_types.CHECK, 0, -1)
        #     if binned_equity >= 10:
        #         if observation["valid_actions"][action_types.CALL.value] == 1:
        #             print("AAA")
        #             return (action_types.CALL, 0, -1)
        #         elif binned_equity >= 14:
        #             print("BBB")
        #             return (action_types.RAISE, observation["min_raise"], -1)
        #     # if observation["valid_actions"][action_types.CALL.value] == 1:
        if action[0].value == action_types.DISCARD.value:
            if observation["valid_actions"][action_types.CHECK.value] == 1:
                print(f"DISCARDING on {street}, {binned_equity}, {flush_number}, {binned_pot_odds} - CHECK was possible ({observation["my_bet"]})")
                return (action_types.CHECK, 0, -1)
        if action[0].value == action_types.RAISE.value:
            if binned_equity < 8:
                print(f"WTF - Raising with {street}, {binned_equity}, {flush_number}, {binned_pot_odds} - raised amount {action[1]} - {observation["valid_actions"][action_types.CHECK.value]}, {observation["valid_actions"][action_types.CALL.value]}")
                if observation["valid_actions"][action_types.CHECK.value] == 1:
                    return (action_types.CHECK, 0, -1)
                elif observation["valid_actions"][action_types.CALL.value] == 1:
                    return (action_types.CALL, 0, -1)
                else:
                    return (action_types.RAISE, observation["min_raise"], -1)
        return action



###################################################################################################
###################################################################################################
###################################################################################################


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
        num_simulations = 4000
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
            self.logger.info(f"Significant hand completed with reward: {reward} - {observation["my_cards"]} - {observation["community_cards"]} - {observation['opp_discarded_card']} - {observation['opp_drawn_card']}")