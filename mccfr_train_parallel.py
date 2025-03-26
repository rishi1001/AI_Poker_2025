import pickle

import concurrent
from cfr import mccfr, poker_game

if __name__ == "__main__":
    game = poker_game.PokerGame()
    trainer = mccfr.MCCFRTrainer(game)
    num_iterations = 100_000_000
    num_processes = 10
    iter_per_process = num_iterations // num_processes

    # Prepare arguments for each process.
    batch_args = [(game, iter_per_process, None) for _ in range(num_processes)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(mccfr.train_batch, args) for args in batch_args]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    merged_regretSum, merged_strategySum = mccfr.merge_updates(results)

    # avg_strategy = trainer.train(100_000_000, save_strat_sum_every=1_000_000)


    # Compute the average strategy from the aggregated strategy sum.
    average_strategy = {}
    for infoSet, strat_sum in merged_strategySum.items():
        total = sum(strat_sum.values())
        if total > 0:
            average_strategy[infoSet] = {a: strat_sum[a] / total for a in strat_sum}
        else:
            n = len(strat_sum)
            average_strategy[infoSet] = {a: 1.0 / n for a in strat_sum}
    
    with open(f"strat_tables/merged_strategy_{num_iterations}.pkl", "wb") as f:
        pickle.dump(average_strategy, f)

    print("Merged strategy computed for", len(average_strategy), "information sets.")
