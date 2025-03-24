import pickle
from itertools import islice

# Load avg_strategy from the pickle file
with open("avg_strategy_uniform_0_clean.pkl", "rb") as f:
    avg_strategy = pickle.load(f)
    

# Print the first 10 items
for infoSet, strategy in islice(avg_strategy.items(), 10):
    print(f"InfoSet {infoSet}: {strategy}")



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