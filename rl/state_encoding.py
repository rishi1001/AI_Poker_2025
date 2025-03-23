import torch

from gym_env import WrappedEval
from rl.utils import to_one_hot_vector

STATE_DIM = 151 + 2
MAX_PLAYER_BET = 100

hand_evaluator = WrappedEval()

def encode_card(card):
    """
    Encodes a card as a 12-dimensional vector:
      - 1 dimension for whether it's visible
      - 9 dimensions for rank (2, 3, ..., 9, Ace)
      - 3 dimensions for suit (diamonds, hearts, spades)
    If card is invalid (e.g. -1), returns a zero vector.
    """
    if card == -1:
        x =  torch.zeros(13, dtype=torch.float32)
        x[0] = 1.0
        return x

    rank_vector = torch.zeros(9, dtype=torch.float32)
    suit_vector = torch.zeros(3, dtype=torch.float32)
    is_invisible = torch.zeros(1, dtype=torch.float32)
    if card >= 0:
        rank_index = card % 9
        suit_index = card // 9
        rank_vector[rank_index] = 1.0
        suit_vector[suit_index] = 1.0
    result = torch.concatenate([is_invisible, rank_vector, suit_vector])
    return result

def obs_to_tensor(obs):
    """
        obs = {
            "street": self.street,
            "acting_agent": self.acting_agent,
            "my_cards": self.player_cards[player_num],
            "community_cards": self.community_cards[:num_cards_to_reveal] + [-1 for _ in range(5 - num_cards_to_reveal)],
            "my_bet": self.bets[player_num],
            "opp_bet": self.bets[1 - player_num],
            "opp_discarded_card": self.discarded_cards[1 - player_num],
            "opp_drawn_card": self.drawn_cards[1 - player_num],
            "my_discarded_card": self.discarded_cards[player_num],
            "my_drawn_card": self.drawn_cards[player_num],
            "min_raise": self.min_raise,
            "max_raise": self.MAX_PLAYER_BET - max(self.bets),
            "valid_actions": self._get_valid_actions(player_num),
        }
    """
    street = to_one_hot_vector(min(3, obs["street"]), 4)
    # acting_agent = to_one_hot_vector(obs["acting_agent"], 2)
    my_cards = torch.concatenate([encode_card(card) for card in obs["my_cards"]])
    community_cards = torch.concatenate([encode_card(card) for card in obs["community_cards"]])
    my_bet = obs["my_bet"] / MAX_PLAYER_BET
    opp_bet = obs["opp_bet"] / MAX_PLAYER_BET
    opp_discarded_card = encode_card(obs["opp_discarded_card"])
    opp_drawn_card = encode_card(obs["opp_drawn_card"])
    my_discarded_card = encode_card(obs["my_discarded_card"]) if "my_discarded_card" in obs else encode_card(-1)
    my_drawn_card = encode_card(obs["my_drawn_card"]) if "my_drawn_card" in obs else encode_card(-1)
    min_raise = obs["min_raise"] / MAX_PLAYER_BET
    max_raise = obs["max_raise"] / MAX_PLAYER_BET

    pocket_pair = obs["my_cards"][0] // 9 == obs["my_cards"][1] // 9
    pocket_suit = obs["my_cards"][0] // 9 == obs["my_cards"][1] // 9

    

    return torch.concatenate([
        street,
        # acting_agent,
        my_cards,
        community_cards,
        torch.tensor([my_bet]),
        torch.tensor([opp_bet]),
        opp_discarded_card,
        opp_drawn_card,
        my_discarded_card,
        my_drawn_card,
        torch.tensor([min_raise]),
        torch.tensor([max_raise]),
        torch.tensor([pocket_pair]),
        torch.tensor([pocket_suit]),
    ])