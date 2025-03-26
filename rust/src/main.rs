// mod abstract_game;
// mod mccfr;
// mod poker_game;

// use mccfr::MCCFRTrainer;
// use poker_game::PokerGame;
// use serde_pickle;
// use std::fs;
// use std::path::Path;

// fn main() {
//     // Create a poker game instance.
//     let game = PokerGame::new(1, 100);
//     let mut trainer = MCCFRTrainer::new(game);

//     // Train for a (possibly large) number of iterations.
//     // (Here we use a lower iteration count for testing.)
//     let avg_strategy = trainer.train(1_000_000_000, 100_000_000, None);

//     //     // Save the average strategy to a file.
//     //     let path = Path::new("strat_tables/avg_strategy_all.json");
//     //     if let Some(parent) = path.parent() {
//     //         fs::create_dir_all(parent).expect("Failed to create directories");
//     //     }
//     //     let file = fs::File::create(path).expect("Failed to create file");
//     //     serde_json::to_writer(file, &avg_strategy).expect("Failed to write avg_strategy");

//     // Save the average strategy to a file using serde_pickle.
//     let path = Path::new("strat_tables/avg_strategy_all.pkl");
//     if let Some(parent) = path.parent() {
//         fs::create_dir_all(parent).expect("Failed to create directories");
//     }
//     let mut file = fs::File::create(path).expect("Failed to create file");
//     serde_pickle::to_writer(&mut file, &avg_strategy, Default::default())
//         .expect("Failed to write avg_strategy");
// }


mod abstract_game;
mod mccfr;
mod poker_game;

use mccfr::{MCCFRTrainer, merge_updates};
use poker_game::PokerGame;
use serde_pickle;
// use std::collections::HashMap;
use hashbrown::HashMap;
use std::io::Write;
use std::time::Instant;
use std::{fs, time};
use std::path::Path;
use std::thread;
use chrono::Local;
use rmp_serde::to_vec;


fn main() {
    // Number of parallel batches. You can set this to the number of CPU cores (using num_cpus::get()).
    let num_batches = 10;
    let iterations_per_batch = 1_000_000;
    
    let mut updates = Vec::new();

    loop {

        // Spawn a thread for each training batch.
        let mut handles = Vec::new();
        for _ in 0..num_batches {
            // Each thread creates its own game instance.
            let game = PokerGame::new(1, 100);
            let handle = thread::spawn(move || {
                let mut trainer = MCCFRTrainer::new(game);
                // Use train_strategy_sum; the second argument (save_strat_sum_every) is set to 0 since we don't save intermediate files.
                trainer.train_strategy_sum(iterations_per_batch, 0, None)
            });
            handles.push(handle);
        }

        // Collect the (regret_sum, strategy_sum) updates from all threads.
        for handle in handles {
            let update = handle.join().expect("Thread panicked");
            updates.push(update);
        }

        // Merge the updates from all batches.
        let (merged_regret_sum, merged_strategy_sum) = merge_updates(&updates);

        // Compute average strategy from the merged strategy sums.
        let mut average_strategy: HashMap<String, HashMap<String, f64>> = HashMap::new();
        for (info_set, strat_sum) in merged_strategy_sum.iter() {
            let total: f64 = strat_sum.values().sum();
            if total > 0.0 {
                let strat: HashMap<String, f64> = strat_sum
                    .iter()
                    .map(|(a, v)| (a.clone(), v / total))
                    .collect();
                average_strategy.insert(info_set.clone(), strat);
            } else {
                let n = strat_sum.len() as f64;
                let strat: HashMap<String, f64> = strat_sum
                    .iter()
                    .map(|(a, _)| (a.clone(), 1.0 / n))
                    .collect();
                average_strategy.insert(info_set.clone(), strat);
            }
        }

        let number_of_info_sets = merged_strategy_sum.len();

        println!("Number of infosets (combined): {}", number_of_info_sets);

        // Save the average strategy using serde_pickle.
        // Get the current local time.
        let now = Local::now();
        
        // Format the time as a string. You can adjust the format string as needed.
        let formatted_time = now.format("%Y%m%d_%H%M%S").to_string();
        let name = format!("strat_tables/avg_strategy_merged_{}.msgpack", formatted_time);
        let path = Path::new(&name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("Failed to create directories");
        }
        let mut file = fs::File::create(path).expect("Failed to create file");
        println!("Started writing avg_strategy");
        // serde_pickle::to_writer(&mut file, &average_strategy, Default::default())
        //     .expect("Failed to write avg_strategy");
        let encoded = to_vec(&average_strategy).unwrap();
        file.write_all(&encoded).unwrap();
        println!("Finished writing avg_strategy");

        // Save the strategy sum using serde_pickle.
        let name = format!("strat_tables/strategy_sum_merged_{}.msgpack", formatted_time);
        let path = Path::new(&name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("Failed to create directories");
        }
        let mut file = fs::File::create(path).expect("Failed to create file");
        println!("Started writing strategy_sum");
        // serde_pickle::to_writer(&mut file, &merged_strategy_sum, Default::default())
        //     .expect("Failed to write strategy_sum");
        let encoded = to_vec(&merged_strategy_sum).unwrap();
        file.write_all(&encoded).unwrap();
        println!("Finished writing strategy_sum");
    }
}
