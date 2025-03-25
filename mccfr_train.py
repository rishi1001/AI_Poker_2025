import pickle
from cfr import mccfr, poker_game

if __name__ == "__main__":
    game = poker_game.PokerGame()
    trainer = mccfr.MCCFRTrainer(game)
    avg_strategy = trainer.train(100_000_000, save_strat_sum_every=1_000_000)

    # Save avg_strategy to a file using pickle
    with open("strat_tables/avg_strategy_all.pkl", "wb") as f:
        pickle.dump(avg_strategy, f)

    # Now print out the strategies for each info set
    # for infoSet, strategy in avg_strategy.items():
    #     print(f"InfoSet {infoSet}: {strategy}")
