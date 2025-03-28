import pickle
from itertools import islice
import time
# import msgpack

from cfr import poker_game2

# Load avg_strategy from the pickle file
with open("submission/merged_avg_strategy.pkl", "rb") as f:
    avg_strategy = pickle.load(f)


for i, d in enumerate(avg_strategy):
    if d is None: continue
    infoset = poker_game2.decode_infoset_int(i)
    a = poker_game2.pretty_action_list(d)
    print(f"binned_equity: {infoset[0]}, valid_actions_number: {infoset[1]}, binned_pot_odds: {infoset[2]}", a)
print(len(avg_strategy))
# valid_actions = set()

# # Print the items in street 0, for player 0, where community cards are all -1
# for infoSet, strategy in avg_strategy.items():
#     x = infoSet.split('_')
#     valid_actions.add(x[7])
#     if x[0] == '1': continue
#     if x[5] != "-1,-1,-1,-1,-1": continue
#     if x[6] != "1": continue
#     print(f"InfoSet {infoSet}: {strategy}")

# print(f"Number of infosets: {len(avg_strategy.keys())}")

# print(f"all possible values of value actions: {valid_actions}")
