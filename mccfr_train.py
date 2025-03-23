from cfr import mccfr, poker_game


if __name__ == "__main__":
    game = poker_game.PokerGame()
    trainer = mccfr.MCCFRTrainer(game)
    avg_strategy = trainer.train(10_000)
    for infoSet, strategy in avg_strategy.items():
        print(f"InfoSet {infoSet}: {strategy}")