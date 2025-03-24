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
        return compute_information_set(obs)
        # if obs["street"] == 0:
        #     # acting_agent: just a byte
        #     # my_hand -> (int, int) -> (sorted) -> turn that to a single integer:  each card is 5 bits: (c1 << 5) + c2
        #     # my_discarded_card
        #     # opp's discarded_card
        #     # opp's drawn_card
        #     # my_bet
        #     # opp's bet
        #     # min_raise
        #     # max_raise
        #     # valid actions ([1, 1, 1, 1, 1] encoded as a byte)
        # elif obs["street"] == 1:
        #     # same as above, plus
        #     # first three community cards -> (int, int, int) -> (sorted) -> turn that to a 128-bit integer
        # elif obs["street"] == 2:
        #     pass
        # elif obs["street"] == 3:
        #     pass
        # else:
        #     raise "wtf - invalid street"
        # return str(obs)

    def get_actions(self, state):
        acting = state["acting_agent"]
        valid = self._get_valid_actions(state, acting)
        actions = []
        # FOLD
        if valid[ActionType.FOLD.value]:
            actions.append((ActionType.FOLD.value, 0, -1))
        # CHECK
        if valid[ActionType.CHECK.value]:
            actions.append((ActionType.CHECK.value, 0, -1))
        # CALL
        if valid[ActionType.CALL.value]:
            actions.append((ActionType.CALL.value, 0, -1))
        # DISCARD (two options: discard card 0 or 1)
        if valid[ActionType.DISCARD.value]:
            actions.append((ActionType.DISCARD.value, 0, 0))
            actions.append((ActionType.DISCARD.value, 0, 1))
        # RAISE: include options for min, max, and if possible intermediate values.
        if valid[ActionType.RAISE.value]:
            obs = self._get_single_player_obs(state, acting)
            min_raise = obs["min_raise"]
            max_raise = obs["max_raise"]
            pot = state["bets"][0] + state["bets"][1]
            actions.append((ActionType.RAISE.value, min_raise, -1))
            actions.append((ActionType.RAISE.value, max_raise, -1))
            if min_raise < pot < max_raise:
                actions.append((ActionType.RAISE.value, pot, -1))
            if min_raise < pot // 2 < max_raise:
                actions.append((ActionType.RAISE.value, pot // 2, -1))
        return actions

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
    def apply_action(self, state, action):
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
        p0_cards = [self.int_to_card(c) for c in state["player_cards"][0] if c != -1]
        p1_cards = [self.int_to_card(c) for c in state["player_cards"][1] if c != -1]
        score0 = self.evaluator.evaluate(p0_cards, board)
        score1 = self.evaluator.evaluate(p1_cards, board)
        if score0 == score1:
            return -1  # Tie
        elif score1 < score0:
            return 1
        else:
            return 0

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

    i_discarded = obs["my_discarded_card"]
    opp_discarded = obs["opp_discarded_card"]

    return f"{player}_{my_cards_sorted}_{community_cards_sorted}_{my_bet_binned}_{opp_bet_binned}_{min_raise_binned}_{max_raise_binned}_{valid_actions}_{i_discarded}_{opp_discarded}"
# import enum

# import torch
# import numpy as np
# import gym_env
# from cfr.mccfr import AbstractGame

# class PokerGame(AbstractGame):
#     def __init__(self):
#         pass

#     def _run_gym_env(self, state, ret_env=False):
#         actions = state["actions"]
#         seed = state["seed"]
#         env = gym_env.PokerEnv()
#         np.random.seed(seed)
#         (obs0, obs1), info = env.reset(seed=seed)
#         terminated = False
#         truncated = False
#         reward0 = reward1 = 0
#         info = {}
#         for action in actions:
#             (obs0, obs1), (reward0, reward1), terminated, truncated, info = env.step(action=action)

#         if ret_env:
#             return (obs0, obs1), (reward0, reward1), terminated, truncated, info, env
#         return (obs0, obs1), (reward0, reward1), terminated, truncated, info
    
#     def get_initial_state(self):
#         """Return the initial state of the game."""
#         return {
#             "actions": [],
#             "seed": torch.randint(0, 1_000_000_000, (1,)).item(),
#         }

#     def is_terminal(self, state):
#         """Return True if state is terminal."""
#         (obs0, obs1), (reward0, reward1), terminated, truncated, info = self._run_gym_env(state)
#         return terminated

#     def get_utility(self, state):
#         """
#         Return a dictionary mapping player IDs to their utility
#         at the terminal state.
#         """
#         (obs0, obs1), (reward0, reward1), terminated, truncated, info = self._run_gym_env(state)
#         return {0: reward0, 1: reward1}

#     def is_chance_node(self, state):
#         """Return True if the state is a chance node (e.g. a random event)."""
#         return False

#     def sample_chance_action(self, state):
#         """For chance nodes, sample and return an action (or outcome)."""
#         raise NotImplementedError

#     def get_current_player(self, state):
#         """Return the player (or chance) whose turn it is to act."""
#         (obs0, obs1), (reward0, reward1), terminated, truncated, info = self._run_gym_env(state)
#         return obs0["acting_agent"]

#     def get_information_set(self, state, player):
#         """
#         Return a key (e.g. a string) that uniquely represents the player’s information set
#         in this state.
#         """
#         (obs0, obs1), (reward0, reward1), terminated, truncated, info = self._run_gym_env(state)
#         # player = obs0["acting_agent"] if player == 0 else obs1["acting_agent"]
#         # hand = obs0["my_cards"] if player == 0 else obs1["my_cards"]
#         # hand_str = f"({hand[0]}, {hand[1]})"
#         # return f"{player}_{hand_str}_{state["actions"].join('->')}"
#         info_set = str(obs0 if player == 0 else obs1)
#         return info_set

#     def get_actions(self, state):
#         """
#         Return a list of actions available at the given state.
#         This allows the set of actions to vary with state.
#         """
#         (obs0, obs1), (reward0, reward1), terminated, truncated, info = self._run_gym_env(state)
#         legal_actions = obs0["valid_actions"] if obs0["acting_agent"] == 0 else obs1["valid_actions"]
#         tuple_actions = []
#         if legal_actions[gym_env.PokerEnv.ActionType.FOLD.value] == 1:
#             tuple_actions.append((gym_env.PokerEnv.ActionType.FOLD.value, 0, -1))
#         if legal_actions[gym_env.PokerEnv.ActionType.CHECK.value] == 1:
#             tuple_actions.append((gym_env.PokerEnv.ActionType.CHECK.value, 0, -1))
#         if legal_actions[gym_env.PokerEnv.ActionType.CALL.value] == 1:
#             tuple_actions.append((gym_env.PokerEnv.ActionType.CALL.value, 0, -1))
#         if legal_actions[gym_env.PokerEnv.ActionType.DISCARD.value] == 1:
#             tuple_actions.append((gym_env.PokerEnv.ActionType.DISCARD.value, 0, 0))
#             tuple_actions.append((gym_env.PokerEnv.ActionType.DISCARD.value, 0, 1))
#         if legal_actions[gym_env.PokerEnv.ActionType.RAISE.value] == 1:
#             min_raise = obs0["min_raise"] if obs0["acting_agent"] == 0 else obs1["min_raise"]
#             max_raise = obs0["max_raise"] if obs0["acting_agent"] == 0 else obs1["max_raise"]
#             pot = obs0["my_bet"] + obs1["my_bet"]
#             tuple_actions.append((gym_env.PokerEnv.ActionType.RAISE.value, min_raise, -1))
#             tuple_actions.append((gym_env.PokerEnv.ActionType.RAISE.value, max_raise, -1))
#             if min_raise < pot < max_raise:
#                 tuple_actions.append((gym_env.PokerEnv.ActionType.RAISE.value, pot, -1))
#             if min_raise < pot//2 < max_raise:
#                 tuple_actions.append((gym_env.PokerEnv.ActionType.RAISE.value, pot//2, -1))

#         # print("tuple_actions", tuple_actions)

#         return tuple_actions


#     def apply_action(self, state, action):
#         """Return the new state after the given action is applied."""
#         (obs0, obs1), (reward0, reward1), terminated, truncated, info, env = self._run_gym_env(state, True)

#         (obs0, obs1), (reward0, reward1), terminated, truncated, info = env.step(action)
#         return {"actions": state["actions"] + [action], "seed": state["seed"]}


#     def get_players(self):
#         """Return a list of players in the game."""
#         return [0, 1]