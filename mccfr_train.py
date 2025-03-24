import pickle
from cfr import mccfr, poker_game

if __name__ == "__main__":
    game = poker_game.PokerGame()
    trainer = mccfr.MCCFRTrainer(game)
    avg_strategy = trainer.train(100_000_000)

    # Save avg_strategy to a file using pickle
    # with open("avg_strategy_all.pkl", "wb") as f:
    #     pickle.dump(avg_strategy, f)

    # 1. filter to keep only infoset that contain 'street': 0
    street_0_avg_strategy = {}
    for infoSet, strategy in avg_strategy.items():
        if "'street': 0" in infoSet:
            street_0_avg_strategy[infoSet] = strategy
    print(f"Size of street_0_avg_strategy: {len(street_0_avg_strategy)}")
    with open("avg_strategy_street_0.pkl", "wb") as f:
        pickle.dump(street_0_avg_strategy, f)


    # 2. filter to keep only infoset that contain 'street': 0 or 1
    street_0_1_avg_strategy = street_0_avg_strategy
    for infoSet, strategy in avg_strategy.items():
        if "'street': 1" in infoSet:
            street_0_1_avg_strategy[infoSet] = strategy
    print(f"Size of street_0_1_avg_strategy: {len(street_0_1_avg_strategy)}")
    with open("avg_strategy_street_0_1.pkl", "wb") as f:
        pickle.dump(street_0_1_avg_strategy, f)

    # Now print out the strategies for each info set
    # for infoSet, strategy in avg_strategy.items():
    #     print(f"InfoSet {infoSet}: {strategy}")
