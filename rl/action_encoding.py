import torch
import torch.nn.functional as F
from gym_env import PokerEnv

"""
0 -> FOLD
1 -> CHECK
2 -> CALL
3 -> DISCARD 0
4 -> DISCARD 1
5 -> RAISE 1
6 -> RAISE 2
...
103 -> RAISE 99
104 -> RAISE 100
"""

ACTION_DIM = 105

def action_int_to_action_tuple(action_int):
    raise_amount = 0
    card_to_discard = -1
    if action_int == 0:
        action_type = PokerEnv.ActionType.FOLD.value
    elif action_int == 1:
        action_type = PokerEnv.ActionType.CHECK.value
    elif action_int == 2:
        action_type = PokerEnv.ActionType.CALL.value
    elif action_int == 3:
        action_type = PokerEnv.ActionType.DISCARD.value
        card_to_discard = 0
    elif action_int == 4:
        action_type = PokerEnv.ActionType.DISCARD.value
        card_to_discard = 1
    else:
        action_type = PokerEnv.ActionType.RAISE.value
        raise_amount = action_int - 4

    return (action_type, raise_amount, card_to_discard)

def compute_valid_actions_mask(valid_actions, min_raise, max_raise):
    # put result[i] = -inf if action i is not valid
    # and 0 otherwise
    result = torch.zeros(ACTION_DIM)
    if valid_actions[PokerEnv.ActionType.CHECK.value] == 0:
        result[1] = -float("inf")
    if valid_actions[PokerEnv.ActionType.CALL.value] == 0:
        result[2] = -float("inf")
    if valid_actions[PokerEnv.ActionType.DISCARD.value] == 0:
        result[3] = -float("inf")
        result[4] = -float("inf")
    if valid_actions[PokerEnv.ActionType.RAISE.value] == 0:
        for i in range(5, ACTION_DIM):
            result[i] = -float("inf")
    else:
        for i in range(5, ACTION_DIM):
            if i - 4 < min_raise or i - 4 > max_raise:
                result[i] = -float("inf")
    return result

def sample_action(action_scores, valid_actions_mask):
    action_scores = action_scores + valid_actions_mask
    # softmax the scores
    action_probs = F.softmax(action_scores, dim=0)
    # sample an action from the distribution
    action = torch.multinomial(action_probs, 1).item()
    return action

def get_random_action(valid_actions_mask):
    same_scores = torch.ones(ACTION_DIM)
    return sample_action(same_scores, valid_actions_mask)