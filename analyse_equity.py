import pickle
import numpy as np
import random
from treys import Evaluator, Card
import sys
# sys.path.append("submission")
from gym_env import PokerEnv, WrappedEval
int_to_card = PokerEnv.int_to_card
from tqdm import tqdm

def get_equity():
    # Calculate equity through Monte Carlo simulation like in the original code
    # shown_cards = my_cards + (community_cards or []) + \
    #                 ([opp_discarded_card] if opp_discarded_card != -1 else []) + \
    #                 ([opp_drawn_card] if opp_drawn_card != -1 else [])
    my_cards = random.sample(range(27), 2)
    
    shown_cards = my_cards
    community_cards = []
    opp_drawn_card = []



    # Cards that are not shown
    non_shown_cards = [i for i in range(27) if i not in shown_cards]

    # Define evaluation function (equivalent to evaluate_hand in original code)
    evaluator = WrappedEval()

    def evaluate_hand(cards):
            my_cards, opp_cards, community_cards = cards
            my_cards = list(map(int_to_card, my_cards))
            opp_cards = list(map(int_to_card, opp_cards))
            community_cards = list(map(int_to_card, community_cards))
            my_hand_rank = evaluator.evaluate(my_cards, community_cards)
            opp_hand_rank = evaluator.evaluate(opp_cards, community_cards)
            return my_hand_rank < opp_hand_rank

        # Run Monte Carlo simulation
    num_simulations = 2000
    wins = sum(
        evaluate_hand((my_cards, opp_drawn_card + drawn_cards[: 2 - len(opp_drawn_card)], community_cards + drawn_cards[2 - len(opp_drawn_card) :]))
        for _ in range(num_simulations)
        if (drawn_cards := random.sample(non_shown_cards, 7 - len(community_cards) - len(opp_drawn_card)))
    )
    equity = wins / num_simulations
    binned_equity = int(equity * 8)
    
    return equity, binned_equity


def create_equal_frequency_bins(equity_list, num_bins=16):
    """Create bins with approximately equal frequencies"""
    # Sort the equity values
    sorted_equity = np.sort(equity_list)
    
    # Calculate the number of samples per bin
    n = len(sorted_equity)
    samples_per_bin = n // (num_bins)
    
    # Create bin edges
    bin_edges = []  # Start with 0
    
    for i in range(1, num_bins):
        idx = i * samples_per_bin
        if idx < n:
            bin_edges.append(sorted_equity[idx])
    
    # bin_edges.append(1.0)  # End with 1
    
    # Ensure bin edges are strictly increasing
    bin_edges = sorted(list(set(bin_edges)))
    
    return bin_edges

if __name__ == "__main__":
    
    # equity_list = []
    # binned_equity_list = []
    
    # for i in tqdm(range(10000)):
    #     equity, binned_equity = get_equity()
    #     equity_list.append(equity)
    #     binned_equity_list.append(binned_equity)
    
    # print(f"Mean Equity: {np.mean(equity_list)}, Mean Binned Equity: {np.mean(binned_equity_list)}")
    
    # # plot equity distribution
    # import matplotlib.pyplot as plt
    # plt.hist(equity_list, bins=30)
    # plt.xlabel("Equity")
    # plt.ylabel("Frequency")
    # plt.title("Equity Distribution")
    # plt.savefig("equity_distribution.png")
    
    # # plot binned equity distribution
    # plt.hist(binned_equity_list, bins=8)
    # plt.xlabel("Binned Equity")
    # plt.ylabel("Frequency")
    # plt.title("Binned Equity Distribution")
    # plt.savefig("binned_equity_distribution.png")
    
    # # plot equity curve
    # equity_list = sorted(equity_list)
    # plt.plot(equity_list)
    # plt.xlabel("Simulation Number")
    # plt.ylabel("Equity")
    # plt.title("Equity Curve")
    # plt.savefig("equity_curve.png")
    
    # # save equity data
    # with open("equity_data.pkl", "wb") as f:
    #     pickle.dump(equity_list, f)
    
    # load equity data
    with open("equity_data.pkl", "rb") as f:
        equity_list = pickle.load(f)
        
    # create equal frequency bins
    bin_edges = create_equal_frequency_bins(equity_list, num_bins=16)
    print("len(bin_edges):", len(bin_edges))
    print(f"Bin Edges: {bin_edges}")
    
    # equity, binned_equity = get_equity()
    # print(f"Equity: {equity}, Binned Equity: {binned_equity}")