import pickle
from itertools import islice
import time

# Load avg_strategy from the pickle file
with open("strat_tables/avg_strategy_all_10m.pkl", "rb") as f:
    avg_strategy = pickle.load(f)
    

# Print the items in street 0, for player 0, where community cards are all -1
for infoSet, strategy in avg_strategy.items():
    x = infoSet.split('_')
    if x[0] == '1': continue
    if x[5] != "-1,-1,-1": continue
    print(f"InfoSet {infoSet}: {strategy}")

print(f"Number of infosets: {len(avg_strategy.keys())}")

time.sleep(5)

with open("strat_sum_9000000.pkl", "rb") as f:
    strat_sum = pickle.load(f)

for infoSet, strat_sum in strat_sum.items():
    x = infoSet.split('_')
    if x[0] == '1': continue
    if x[5] != "-1,-1,-1": continue
    print(f"InfoSet {infoSet}: {strat_sum}")

# total = 0
# uniform = 0

# new_avg_strategy = {}

# # Print the first 10 items
# for infoSet, strategy in avg_strategy.items():
#     total += 1
#     uniform += 1 if all(v == 1.0 / len(strategy) for v in strategy.values()) else 0

#     if not all(v == 1.0 / len(strategy) for v in strategy.values()):
#         new_avg_strategy[infoSet] = strategy

# print(f"Total: {total}")
# print(f"Uniform: {uniform}")

# with open("avg_strategy_uniform_0_clean.pkl", "wb") as f:
#     pickle.dump(new_avg_strategy, f)