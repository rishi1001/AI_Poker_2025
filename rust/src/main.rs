mod abstract_game;
mod mccfr;
mod poker_game;

use std::fs;
use std::path::Path;

use mccfr::MCCFR;
use poker_game::decode_infoset_int;
use poker_game::NiceInfoSet;
// your MCCFR solver module
use poker_game::pretty_action_list;
use poker_game::PokerGame; // your PokerGame module // your helper functions for decoding & pretty printing

// For serialization we use bincode (add it to Cargo.toml)
use serde_pickle;

fn main() {
    // Create the output directory if it does not exist.
    let output_dir = "strat_tables";
    if !Path::new(output_dir).exists() {
        fs::create_dir(output_dir).expect("Failed to create output directory");
    }

    // Instantiate the game.
    // (Make sure your PokerGame implementation uses integer information sets.)
    let game_instance = PokerGame::new(1, 100);
    let num_info_sets = 31_711_680; // total number of information sets
    let num_actions = 9; // maximum number of actions per info set

    // Create the MCCFR solver.
    let mut mccfr_solver = MCCFR::new(game_instance, num_info_sets, num_actions);

    println!("Starting");

    // Run iterations.
    // Here we run 1,000,000 iterations; in each iteration we update for both players.
    for iteration in 0..1_000_000 {
        for player in [0, 1].iter() {
            mccfr_solver.run_iteration(*player);
        }

        // Every 10 iterations, save the strategy tables and print selected info.
        if iteration % 1000 == 0 {
            // Compute the average strategy (a vector indexed by info set).
            let avg_strategy: Vec<Option<Vec<f64>>> = mccfr_solver.compute_average_strategy();
            let avg_strategy: Vec<_> = avg_strategy
                .into_iter()
                .map(|opt| opt.unwrap_or_else(|| vec![]))
                .collect();

            let filename_avg = format!("{}/avg_strategy_{}.pkl", output_dir, iteration);
            let filename_cum = format!("{}/cumulative_strategy_{}.pkl", output_dir, iteration);

            // Serialize using serde_pickle (ensure your types implement Serialize)
            let avg_data = serde_pickle::to_vec(&avg_strategy, Default::default())
                .expect("Serialization failed for avg_strategy");
            fs::write(&filename_avg, avg_data).expect("Failed writing avg_strategy file");

            let cum_data =
                serde_pickle::to_vec(&mccfr_solver.cumulative_strategy, Default::default())
                    .expect("Serialization failed for cumulative_strategy");
            fs::write(&filename_cum, cum_data).expect("Failed writing cumulative_strategy file");

            // For each modified information set, decode and print if the community tuple equals (0,0,0,0,0).
            for &infoset in mccfr_solver.modified_infosets.iter() {
                let nice_info_set: NiceInfoSet = decode_infoset_int(infoset);
                // In this example we check if the community field equals [0, 0, 0, 0, 0].
                if nice_info_set.community != vec![0, 0, 0, 0, 0] {
                    continue;
                }
                let nice_action_list = pretty_action_list(&avg_strategy[infoset]);
                println!("{:?}: {}", nice_info_set, nice_action_list);
            }
            println!("Iteration {} finished", iteration);
        }
    }

    // After iterations finish, print the overall average strategy for the matching info sets.
    let avg_strategy = mccfr_solver.compute_average_strategy();
    let avg_strategy: Vec<_> = avg_strategy
        .into_iter()
        .map(|opt| opt.unwrap_or_else(|| vec![]))
        .collect();
    println!("Average Strategy (Info Set index -> Action Probabilities):");
    for &infoset in mccfr_solver.modified_infosets.iter() {
        let nice_info_set: NiceInfoSet = decode_infoset_int(infoset);
        if nice_info_set.community != vec![0, 0, 0, 0, 0] {
            continue;
        }
        let nice_action_list = pretty_action_list(&avg_strategy[infoset]);
        println!("{:?}: {}", nice_info_set, nice_action_list);
    }
}
