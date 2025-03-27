
import torch
from rl import state_encoding

INFO_DIM = 117


def encode_letter_card(card):
    number = card[0]
    suit = card[1]
    if number == "A":
        number = "10"
    number = int(number) - 2

    suit = "dhs".index(suit)

    return suit * 9 + number

def info_to_tensor(info):
    """
    e.g. {'player_0_cards': ['8s', '3s'], 'player_1_cards': ['4h', '2d'], 'community_cards': [], 'hand_number': 38}
    """
    player_0_cards = list(map(encode_letter_card, info["player_0_cards"]))
    player_1_cards = list(map(encode_letter_card, info["player_1_cards"]))
    community_cards = list(map(encode_letter_card, info["community_cards"]))
    if len(community_cards) < 5:
        community_cards += [-1 for _ in range(5 - len(community_cards))]

    # encode to tensors
    p0cards = torch.cat([state_encoding.encode_card(card) for card in player_0_cards])
    p1cards = torch.cat([state_encoding.encode_card(card) for card in player_1_cards])
    commcards = torch.cat([state_encoding.encode_card(card) for card in community_cards])

    return torch.cat([p0cards, p1cards, commcards])