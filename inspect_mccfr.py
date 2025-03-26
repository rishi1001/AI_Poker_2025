import pickle
from itertools import islice
import time
import msgpack

# Load avg_strategy from the pickle file
# with open("strat_tables/avg_strategy_all.pkl", "rb") as f:
#     avg_strategy = pickle.load(f)

# Open and read the file
with open("rust/strat_tables/avg_strategy_merged_20250326_082536.msgpack", "rb") as f:
    # Unpack with raw=False to convert byte strings to regular strings
    avg_strategy = msgpack.unpackb(f.read(), raw=False)    

valid_actions = set()

# Print the items in street 0, for player 0, where community cards are all -1
for infoSet, strategy in avg_strategy.items():
    x = infoSet.split('_')
    valid_actions.add(x[7])
    if x[0] == '1': continue
    if x[5] != "-1,-1,-1,-1,-1": continue
    if x[6] != "1": continue
    print(f"InfoSet {infoSet}: {strategy}")

print(f"Number of infosets: {len(avg_strategy.keys())}")

print(f"all possible values of value actions: {valid_actions}")
