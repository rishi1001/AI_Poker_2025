mod abstract_game;
mod mccfr;
mod poker_game;

use poker_game::PokerGame;
use mccfr::MCCFRTrainer;
use std::fs;
use std::path::Path;
use serde_json;
use serde_pickle;


fn main() {
    // Create a poker game instance.
    let game = PokerGame::new(1, 100);
    let mut trainer = MCCFRTrainer::new(game);
    
    // Train for a (possibly large) number of iterations.
    // (Here we use a lower iteration count for testing.)
    let avg_strategy = trainer.train(1_000_000, 10_000_000, None);
    
//     // Save the average strategy to a file.
//     let path = Path::new("strat_tables/avg_strategy_all.json");
//     if let Some(parent) = path.parent() {
//         fs::create_dir_all(parent).expect("Failed to create directories");
//     }
//     let file = fs::File::create(path).expect("Failed to create file");
//     serde_json::to_writer(file, &avg_strategy).expect("Failed to write avg_strategy");

    // Save the average strategy to a file using serde_pickle.
    let path = Path::new("strat_tables/avg_strategy_all.pkl");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("Failed to create directories");
    }
    let mut file = fs::File::create(path).expect("Failed to create file");
    serde_pickle::to_writer(&mut file, &avg_strategy, Default::default())
        .expect("Failed to write avg_strategy");
}
