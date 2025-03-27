mod abstract_game;
mod mccfr;
mod poker_game;

use mccfr::MCCFR;
use poker_game::{
    decode_infoset_int, NiceInfoSet, pretty_action_list, PokerGame, PokerState,
};
use serde_pickle;
use std::fs;
use std::path::Path;
use std::thread;

const NUM_INFO_SETS: usize = 360; // total number of information sets
const NUM_ACTIONS: usize = 9; // maximum number of actions per info set
const ITERATIONS: usize = 50_000;

fn main() {
    // Create the output directory if it does not exist.
    let output_dir = "strat_tables";
    if !Path::new(output_dir).exists() {
        fs::create_dir(output_dir).expect("Failed to create output directory");
    }

    // Spawn 8 threads. Each thread will run its own MCCFR solver and, after ITERATIONS,
    // save its cumulative strategy into a file named "cumulative_strategy_{thread_id}.pkl".
    let mut handles = Vec::new();
    for thread_id in 0..8 {
        let output_dir = output_dir.to_string();
        let handle = thread::spawn(move || {
            // Each thread instantiates its own game and MCCFR solver.
            let game_instance = PokerGame::new(1, 100);
            let mut solver = MCCFR::new(game_instance, NUM_INFO_SETS, NUM_ACTIONS);

            // Run iterations: update for both players in each iteration.
            for iteration in 0..ITERATIONS {
                for player in [0, 1].iter() {
                    solver.run_iteration(*player);
                }

                println!("Iteration {} finished", iteration);
                if iteration % 10 == 0 {
                    // Save only the cumulative strategy.
                    let filename = format!("{}/cumulative_strategy_{}.pkl", output_dir, thread_id);
                    let data =
                        serde_pickle::to_vec(&solver.cumulative_strategy, Default::default())
                            .expect("Serialization failed for cumulative strategy");
                    fs::write(&filename, data).expect(&format!(
                        "Failed writing cumulative strategy file {}",
                        filename
                    ));

                    
                    if thread_id == 0 {
                        // Compute the average strategy (a vector indexed by info set).
                        let avg_strategy: Vec<Option<Vec<f64>>> = solver.compute_average_strategy();
                        let avg_strategy: Vec<_> = avg_strategy
                            .into_iter()
                            .map(|opt| opt.unwrap_or_else(|| vec![]))
                            .collect();
                        // For each modified information set, decode and print if the community tuple equals (0,0,0,0,0).
                        for &infoset in solver.modified_infosets.iter() {
                            let nice_info_set: NiceInfoSet = decode_infoset_int(infoset);
                            let nice_action_list = pretty_action_list(&avg_strategy[infoset]);
                            println!("{:?}: {}", nice_info_set, nice_action_list);
                        }
                    }
                }
            }

            // Save only the cumulative strategy.
            let filename = format!("{}/cumulative_strategy_{}.pkl", output_dir, thread_id);
            let data =
                serde_pickle::to_vec(&solver.cumulative_strategy, Default::default())
                    .expect("Serialization failed for cumulative strategy");
            fs::write(&filename, data).expect(&format!(
                "Failed writing cumulative strategy file {}",
                filename
            ));

            // Compute the average strategy (a vector indexed by info set).
            let avg_strategy: Vec<Option<Vec<f64>>> = solver.compute_average_strategy();
            let avg_strategy: Vec<_> = avg_strategy
                .into_iter()
                .map(|opt| opt.unwrap_or_else(|| vec![]))
                .collect();

            println!(
                "Thread {} finished and saved its cumulative strategy.",
                thread_id
            );


        });
        handles.push(handle);
    }

    // Wait for all threads to finish.
    for handle in handles {
        handle.join().unwrap();
    }

    // Merge all cumulative strategies from the folder.
    merge_strategies("strat_tables", NUM_INFO_SETS, NUM_ACTIONS);
}

/// Reads all files in `dir` that start with "cumulative_strategy_" and ending with ".pkl",
/// then merges them into a single cumulative strategy (by summing elementwise).
/// Then computes an average strategy (by normalizing each info set's vector),
/// and writes both merged cumulative and merged average strategies into files.
fn merge_strategies(dir: &str, num_info_sets: usize, num_actions: usize) {
    // Initialize a merged cumulative strategy as a Vec<Option<Vec<f64>>> of zeros.
    let mut merged_cumulative: Vec<Option<Vec<f64>>> = vec![None; num_info_sets];

    // Read each file in the folder that starts with "cumulative_strategy_".
    let entries = fs::read_dir(dir).expect("Failed reading directory");
    let mut count = 0;
    for entry in entries {
        let entry = entry.expect("Error reading entry");
        let filename = entry.file_name();
        let filename_str = filename.to_string_lossy();
        if filename_str.starts_with("cumulative_strategy_") && filename_str.ends_with(".pkl") {
            let data = fs::read(entry.path()).expect("Failed reading file");
            // The file was saved as a Vec<Option<Vec<f64>>>
            let cum_strategy: Vec<Option<Vec<f64>>> =
                serde_pickle::from_slice(&data, Default::default())
                    .expect("Deserialization failed for cumulative strategy file");
            // Merge: for each information set, add the vector (or zeros if None).
            for (i, opt_vec) in cum_strategy.into_iter().enumerate() {
                if opt_vec.is_none() {
                    continue;
                }
                let vec_strategy = opt_vec.unwrap();
                for a in 0..num_actions {
                    if merged_cumulative[i].is_none() {
                        merged_cumulative[i] = Some(vec![0.0; num_actions]);
                    }
                    merged_cumulative[i].as_mut().unwrap()[a] += vec_strategy[a];
                }
            }
            count += 1;
            println!("Merged file: {}", filename_str);
        }
    }
    println!("Merged {} cumulative strategy files.", count);

    // Compute the average strategy from the merged cumulative strategy.
    let merged_avg: Vec<Option<Vec<f64>>> = merged_cumulative
        .iter()
        .map(|action_counts| {
            if action_counts.is_none() {
                return None
            }
            let total: f64 = action_counts.as_ref().unwrap().iter().sum();
            if total > 0.0 {
                Some(action_counts.as_ref().unwrap().iter().map(|&c| c / total).collect())
            } else {
                // Uniform strategy if no counts recorded.
                Some(vec![1.0 / num_actions as f64; num_actions])
            }
        })
        .collect();

    // Save the merged cumulative and average strategies.
    let merged_cumulative_file = format!("{}/merged_cumulative_strategy.pkl", dir);
    let merged_avg_file = format!("{}/merged_avg_strategy.pkl", dir);

    let merged_cum_data = serde_pickle::to_vec(&merged_cumulative, Default::default())
        .expect("Serialization failed for merged cumulative strategy");
    fs::write(&merged_cumulative_file, merged_cum_data)
        .expect("Failed writing merged cumulative strategy file");

    let merged_avg_data = serde_pickle::to_vec(&merged_avg, Default::default())
        .expect("Serialization failed for merged average strategy");
    fs::write(&merged_avg_file, merged_avg_data)
        .expect("Failed writing merged average strategy file");

    println!("Merged cumulative and average strategies saved.");
}

// mod abstract_game;
// mod mccfr;
// mod poker_game;

// use std::fs;
// use std::path::Path;

// use mccfr::MCCFR;
// use poker_game::decode_infoset_int;
// use poker_game::NiceInfoSet;
// // your MCCFR solver module
// use poker_game::pretty_action_list;
// use poker_game::PokerGame; // your PokerGame module // your helper functions for decoding & pretty printing

// // For serialization we use bincode (add it to Cargo.toml)
// use serde_pickle;

// fn main() {
//     // Create the output directory if it does not exist.
//     let output_dir = "strat_tables";
//     if !Path::new(output_dir).exists() {
//         fs::create_dir(output_dir).expect("Failed to create output directory");
//     }

//     // Instantiate the game.
//     // (Make sure your PokerGame implementation uses integer information sets.)
//     let game_instance = PokerGame::new(1, 100);
//     let num_info_sets = 31_711_680; // total number of information sets
//     let num_actions = 9; // maximum number of actions per info set

//     // Create the MCCFR solver.
//     let mut mccfr_solver = MCCFR::new(game_instance, num_info_sets, num_actions);

//     println!("Starting");

//     // Run iterations.
//     // Here we run 1,000,000 iterations; in each iteration we update for both players.
//     for iteration in 0..1_000_000 {
//         for player in [0, 1].iter() {
//             mccfr_solver.run_iteration(*player);
//         }

//         // Every 10 iterations, save the strategy tables and print selected info.
//         if iteration % 1000 == 0 {
//             // Compute the average strategy (a vector indexed by info set).
//             let avg_strategy: Vec<Option<Vec<f64>>> = mccfr_solver.compute_average_strategy();
//             let avg_strategy: Vec<_> = avg_strategy
//                 .into_iter()
//                 .map(|opt| opt.unwrap_or_else(|| vec![]))
//                 .collect();

//             let filename_avg = format!("{}/avg_strategy_{}.pkl", output_dir, iteration);
//             let filename_cum = format!("{}/cumulative_strategy_{}.pkl", output_dir, iteration);

//             // Serialize using serde_pickle (ensure your types implement Serialize)
//             let avg_data = serde_pickle::to_vec(&avg_strategy, Default::default())
//                 .expect("Serialization failed for avg_strategy");
//             fs::write(&filename_avg, avg_data).expect("Failed writing avg_strategy file");

//             let cum_data =
//                 serde_pickle::to_vec(&mccfr_solver.cumulative_strategy, Default::default())
//                     .expect("Serialization failed for cumulative_strategy");
//             fs::write(&filename_cum, cum_data).expect("Failed writing cumulative_strategy file");

//             // For each modified information set, decode and print if the community tuple equals (0,0,0,0,0).
//             for &infoset in mccfr_solver.modified_infosets.iter() {
//                 let nice_info_set: NiceInfoSet = decode_infoset_int(infoset);
//                 // In this example we check if the community field equals [0, 0, 0, 0, 0].
//                 if nice_info_set.community != vec![0, 0, 0, 0, 0] {
//                     continue;
//                 }
//                 let nice_action_list = pretty_action_list(&avg_strategy[infoset]);
//                 println!("{:?}: {}", nice_info_set, nice_action_list);
//             }
//             println!("Iteration {} finished", iteration);
//         }
//     }

//     // After iterations finish, print the overall average strategy for the matching info sets.
//     let avg_strategy = mccfr_solver.compute_average_strategy();
//     let avg_strategy: Vec<_> = avg_strategy
//         .into_iter()
//         .map(|opt| opt.unwrap_or_else(|| vec![]))
//         .collect();
//     println!("Average Strategy (Info Set index -> Action Probabilities):");
//     for &infoset in mccfr_solver.modified_infosets.iter() {
//         let nice_info_set: NiceInfoSet = decode_infoset_int(infoset);
//         if nice_info_set.community != vec![0, 0, 0, 0, 0] {
//             continue;
//         }
//         let nice_action_list = pretty_action_list(&avg_strategy[infoset]);
//         println!("{:?}: {}", nice_info_set, nice_action_list);
//     }
// }
