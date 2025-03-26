# The game instance must now conform to the updated conventions:
# - get_information_set(state) returns an integer.
# - get_actions(state) returns a list of available action indices.
#
# For example, you might implement a version of Matching Pennies or Kuhn Poker that uses integer info sets.
#
# Here we assume game_instance is defined in cfr.poker_game and now uses integer information sets.
import pickle
from cfr import poker_game2
from cfr.mccfr2 import MCCFR
game_instance = poker_game2.PokerGame()  # Ensure this version uses integer info sets.

num_info_sets = 31711680  # Set the total number of information sets.
num_actions = 9      # Set the maximum number of actions per information set

mccfr_solver = MCCFR(game_instance, num_info_sets, num_actions)

print("Starting")

# Run a number of iterations for each player.
for iteration in range(1_000_000):
    for player in [0, 1]:
        mccfr_solver.run_iteration(player)
        # Save avg_strategy to a file using pickle
    avg_strategy = mccfr_solver.compute_average_strategy()

    if iteration % 10 == 0:
        with open(f"strat_tables/avg_strategy_{iteration}.pkl", "wb") as f:
            pickle.dump(avg_strategy, f)
        with open(f"strat_tables/cumulative_strategy_{iteration}.pkl", "wb") as f:
            pickle.dump(mccfr_solver.cumulative_strategy, f)
        for infoset in mccfr_solver.modified_infosets:
            nice_info_set = poker_game2.decode_infoset_int(infoset)
            if nice_info_set[4] != (0, 0, 0, 0, 0): continue
            nice_action_list = poker_game2.pretty_action_list(avg_strategy[infoset])
            print(f"{nice_info_set}: {nice_action_list}")
        print(f"Iteration {iteration} finished")

avg_strategy = mccfr_solver.compute_average_strategy()

print("Average Strategy (Info Set index -> Action Probabilities):")
for infoset in mccfr_solver.modified_infosets:
    nice_info_set = poker_game2.decode_infoset_int(infoset)
    if nice_info_set[4] != (0, 0, 0, 0, 0): continue

    nice_action_list = poker_game2.pretty_action_list(avg_strategy[infoset])
    print(f"{nice_info_set}: {nice_action_list}")
