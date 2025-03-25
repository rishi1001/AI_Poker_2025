use crate::abstract_game::AbstractGame;
use rand::Rng;
// use std::collections::HashMap;
use hashbrown::HashMap;

/// The Monte Carlo CFR trainer. It holds cumulative regret and strategy tables.
pub struct MCCFRTrainer<G: AbstractGame> {
    pub game: G,
    pub regret_sum: HashMap<String, HashMap<String, f64>>,
    pub strategy_sum: HashMap<String, HashMap<String, f64>>,
}

impl<G: AbstractGame> MCCFRTrainer<G> {
    pub fn new(game: G) -> Self {
        Self {
            game,
            regret_sum: HashMap::new(),
            strategy_sum: HashMap::new(),
        }
    }

    pub fn reset(&mut self) {
        self.regret_sum.clear();
        self.strategy_sum.clear();
    }

    /// Given an information set and the available actions, compute the current strategy
    /// (via regret matching) and update the cumulative strategy sum.
    pub fn get_strategy(
        &mut self,
        info_set: &str,
        available_actions: &[String],
        realization_weight: f64,
    ) -> HashMap<String, f64> {
        if !self.regret_sum.contains_key(info_set) {
            let mut map = HashMap::new();
            for a in available_actions {
                map.insert(a.clone(), 0.0);
            }
            self.regret_sum.insert(info_set.to_string(), map);
        }
        let regrets = self.regret_sum.get(info_set).unwrap();
        let mut normalizing_sum = 0.0;
        for a in available_actions {
            normalizing_sum += regrets.get(a).cloned().unwrap_or(0.0).max(0.0);
        }
        let strategy: HashMap<String, f64> = if normalizing_sum > 0.0 {
            available_actions
                .iter()
                .map(|a| {
                    let reg = regrets.get(a).cloned().unwrap_or(0.0);
                    (a.clone(), reg.max(0.0) / normalizing_sum)
                })
                .collect()
        } else {
            let uniform = 1.0 / available_actions.len() as f64;
            available_actions
                .iter()
                .map(|a| (a.clone(), uniform))
                .collect()
        };

        if !self.strategy_sum.contains_key(info_set) {
            let mut map = HashMap::new();
            for a in available_actions {
                map.insert(a.clone(), 0.0);
            }
            self.strategy_sum.insert(info_set.to_string(), map);
        } else {
            let strat_sum = self.strategy_sum.get_mut(info_set).unwrap();
            for a in available_actions {
                strat_sum.entry(a.clone()).or_insert(0.0);
            }
        }
        if let Some(strat_sum) = self.strategy_sum.get_mut(info_set) {
            for a in available_actions {
                let entry = strat_sum.entry(a.clone()).or_insert(0.0);
                *entry += realization_weight * strategy.get(a).unwrap_or(&0.0);
            }
        }
        strategy
    }

    /// Sample an action from a strategy (a mapping from action to probability).
    pub fn sample_action(&self, strategy: &HashMap<String, f64>) -> String {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        let mut cumulative_probability = 0.0;
        for (a, prob) in strategy.iter() {
            cumulative_probability += prob;
            if r < cumulative_probability {
                return a.clone();
            }
        }
        strategy.keys().last().unwrap().clone()
    }

    /// Recursively perform outcome-sampling MCCFR.
    pub fn cfr(
        &mut self,
        state: &G::State,
        reach_probs: &HashMap<G::Player, f64>,
        sample_probs: &HashMap<G::Player, f64>,
    ) -> HashMap<G::Player, f64> {
        if self.game.is_terminal(state) {
            return self.game.get_utility(state);
        }
        if self.game.is_chance_node(state) {
            let action = self.game.sample_chance_action(state);
            let next_state = self.game.apply_action(state, &action);
            return self.cfr(&next_state, reach_probs, sample_probs);
        }
        let current_player = self.game.get_current_player(state);
        let info_set = self.game.get_information_set(state, current_player);
        let available_actions = self.game.get_actions(state);
        let strategy =
            self.get_strategy(&info_set, &available_actions, *reach_probs.get(&current_player).unwrap());
        
        // Outcome sampling: sample one action.
        let sampled_action = self.sample_action(&strategy);
        let new_state = self.game.apply_action(state, &sampled_action);
        let mut new_sample_probs = sample_probs.clone();
        if let Some(prob) = strategy.get(&sampled_action) {
            let current_sp = new_sample_probs.get_mut(&current_player).unwrap();
            *current_sp *= *prob;
        }
        let utilities = self.cfr(&new_state, reach_probs, &new_sample_probs);
        let util_current = *utilities.get(&current_player).unwrap();
        for a in available_actions.clone() {
            let action_util = if a == sampled_action {
                if let Some(prob) = strategy.get(&a) {
                    if *prob != 0.0 { util_current / *prob } else { 0.0 }
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let regret = action_util - util_current;
            if !self.regret_sum.contains_key(&info_set) {
                let mut map = HashMap::new();
                for act in &available_actions {
                    map.insert(act.clone(), 0.0);
                }
                self.regret_sum.insert(info_set.clone(), map);
            }
            if let Some(regret_map) = self.regret_sum.get_mut(&info_set) {
                let entry = regret_map.entry(a.clone()).or_insert(0.0);
                if let Some(sample_prob) = sample_probs.get(&current_player) {
                    *entry += (1.0 / *sample_prob) * regret;
                }
            }
        }
        utilities
    }

    /// Run MCCFR for a given number of iterations, periodically saving the strategy sums.
    pub fn train(
        &mut self,
        iterations: usize,
        save_strat_sum_every: usize,
        custom_initial_state: Option<G::State>,
    ) -> HashMap<String, HashMap<String, f64>> {
        let players = self.game.get_players();
        for i in 0..iterations {
            let mut reach_probs = HashMap::new();
            let mut sample_probs = HashMap::new();
            for p in &players {
                reach_probs.insert(*p, 1.0);
                sample_probs.insert(*p, 1.0);
            }
            let initial_state = match &custom_initial_state {
                Some(state) => state.clone(),
                None => self.game.get_initial_state(),
            };
            self.cfr(&initial_state, &reach_probs, &sample_probs);

            if i % 10000 == 0 {
                println!("Iteration {} - Number of infosets recorded: {}", i, self.strategy_sum.len());
            }
            if i % save_strat_sum_every == 0 {
                let filename = format!("strat_sum_{}.json", i);
                if let Ok(file) = std::fs::File::create(&filename) {
                    let _ = serde_json::to_writer(file, &self.strategy_sum);
                }
            }
        }
        let mut average_strategy = HashMap::new();
        for (info_set, strat_sum) in &self.strategy_sum {
            let total: f64 = strat_sum.values().sum();
            if total > 0.0 {
                let strat: HashMap<String, f64> =
                    strat_sum.iter().map(|(a, v)| (a.clone(), v / total)).collect();
                average_strategy.insert(info_set.clone(), strat);
            } else {
                let n = strat_sum.len() as f64;
                let strat: HashMap<String, f64> =
                    strat_sum.iter().map(|(a, _)| (a.clone(), 1.0 / n)).collect();
                average_strategy.insert(info_set.clone(), strat);
            }
        }
        average_strategy
    }

    /// A variant that returns the raw regret and strategy sums.
    pub fn train_strategy_sum(
        &mut self,
        iterations: usize,
        _save_strat_sum_every: usize,
        custom_initial_state: Option<G::State>,
    ) -> (HashMap<String, HashMap<String, f64>>, HashMap<String, HashMap<String, f64>>) {
        let players = self.game.get_players();
        for i in 0..iterations {
            let mut reach_probs = HashMap::new();
            let mut sample_probs = HashMap::new();
            for p in &players {
                reach_probs.insert(*p, 1.0);
                sample_probs.insert(*p, 1.0);
            }
            let initial_state = match &custom_initial_state {
                Some(state) => state.clone(),
                None => self.game.get_initial_state(),
            };
            self.cfr(&initial_state, &reach_probs, &sample_probs);
            if i % 10000 == 0 && i > 0 {
                println!("Iteration {} - Number of infosets recorded: {}", i, self.strategy_sum.len());
            }
        }
        (self.regret_sum.clone(), self.strategy_sum.clone())
    }
}

/// Merge a list of updates (each a pair of regret and strategy tables) into one.
pub fn merge_updates(
    updates: Vec<(HashMap<String, HashMap<String, f64>>, HashMap<String, HashMap<String, f64>>)>,
) -> (HashMap<String, HashMap<String, f64>>, HashMap<String, HashMap<String, f64>>) {
    let mut merged_regret_sum: HashMap<String, HashMap<String, f64>> = HashMap::new();
    let mut merged_strategy_sum: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for (regret_sum, strategy_sum) in updates {
        for (info_set, action_dict) in regret_sum {
            let entry = merged_regret_sum.entry(info_set).or_insert_with(HashMap::new);
            for (a, val) in action_dict {
                *entry.entry(a).or_insert(0.0) += val;
            }
        }
        for (info_set, action_dict) in strategy_sum {
            let entry = merged_strategy_sum.entry(info_set).or_insert_with(HashMap::new);
            for (a, val) in action_dict {
                *entry.entry(a).or_insert(0.0) += val;
            }
        }
    }
    (merged_regret_sum, merged_strategy_sum)
}
