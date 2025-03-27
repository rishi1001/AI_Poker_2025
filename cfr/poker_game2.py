import enum
import copy
import math
import torch
import numpy as np
from treys import Card, Evaluator

from cfr.abstract_game import AbstractGame

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
            actions.append(0)
        # CHECK
        if valid[ActionType.CHECK.value]:
            actions.append(1)
        # CALL
        if valid[ActionType.CALL.value]:
            actions.append(2)
        # DISCARD (two options: discard card 0 or 1)
        if valid[ActionType.DISCARD.value]:
            actions.append(3)
            actions.append(4)
        # RAISE: include options for min, max, and if possible intermediate values.
        if valid[ActionType.RAISE.value]:
            obs = self._get_single_player_obs(state, acting)
            actions.append(5) # min raise
            actions.append(6) # max raise
            actions.append(7) # pot
            actions.append(8) # half pot
        return actions
    
    def action_int_to_action_tuple(self, state, action_int):
        if action_int == 0:
            return (ActionType.FOLD.value, 0, -1)
        elif action_int == 1:
            return (ActionType.CHECK.value, 0, -1)
        elif action_int == 2:
            return (ActionType.CALL.value, 0, -1)
        elif action_int == 3:
            current_hand = state["player_cards"][state["acting_agent"]]
            lower_card_idx = 0 if current_hand[0] % 9 <= current_hand[1] % 9 else 1
            return (ActionType.DISCARD.value, 0, lower_card_idx)
        elif action_int == 4:
            current_hand = state["player_cards"][state["acting_agent"]]
            higher_card_idx = 0 if current_hand[0] % 9 >= current_hand[1] % 9 else 1
            return (ActionType.DISCARD.value, 0, higher_card_idx)
        elif action_int == 5:
            min_raise = min(state["min_raise"], self.max_player_bet - max(state["bets"]))
            return (ActionType.RAISE.value, min_raise, -1)
        elif action_int == 6:
            return (ActionType.RAISE.value, self.max_player_bet - max(state["bets"]), -1)
        elif action_int == 7:
            max_raise = self.max_player_bet - max(state["bets"])
            min_raise = min(state["min_raise"], max_raise)
            pot = sum(state["bets"])
            safe_bet = max(min_raise, min(max_raise, pot))
            return (ActionType.RAISE.value, safe_bet, -1)
        elif action_int == 8:
            max_raise = self.max_player_bet - max(state["bets"])
            min_raise = min(state["min_raise"], max_raise)
            pot = sum(state["bets"])
            half_pot = pot // 2
            safe_bet = max(min_raise, min(max_raise, half_pot))
            return (ActionType.RAISE.value, safe_bet, -1)
        else:
            raise f"wtf - invalid action_int {action_int}"

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
        action = self.action_int_to_action_tuple(state, action_str)
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
        pot_odds = continuation_cost / pot

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

def decode_infoset_int(infoset):
    radices = (2    , 55                 , 2                             , 3           , 2002                      , 8                   , 3              )
    x = list(decode_fields(infoset, radices))
    x[1] = int_to_tuple_2(x[1])
    x[4] = int_to_tuple_5(x[4])

    VALID_ACTIONS_MAP = [
        "11101",
        "10010",
        "11011",
        "10101",
        "11010",
        "11100",
        "10100",
        "10011",
    ]
    x[5] = VALID_ACTIONS_MAP[x[5]]
    return x

from functools import reduce

def pretty_action_list(action_probabilities):
    f_p = action_probabilities[0]
    ch_p = action_probabilities[1]
    call_p = action_probabilities[2]
    d0_p = action_probabilities[3]
    d1_p = action_probabilities[4]
    r_min_p = action_probabilities[5]
    r_max_p = action_probabilities[6]
    r_pot_p = action_probabilities[7]
    r_hp_p = action_probabilities[8]
    return f"F:{f_p:.3f}|Ch:{ch_p:.3f}|Ca:{call_p:.3f}|D0:{d0_p:.3f}|D1:{d1_p:.3f}|Rmin:{r_min_p:.3f}|Rmax:{r_max_p:.3f}|Rp:{r_pot_p:.3f}|Rhp:{r_hp_p:.3f}"

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
