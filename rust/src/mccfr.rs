use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use std::collections::HashSet;
use std::marker::PhantomData;


use crate::abstract_game::AbstractGame;

/// Compute a strategy using regret matching.
/// For each action in `regrets`, if the action is available and its regret is positive,
/// it is taken into account; otherwise it is set to 0. Then the strategy is normalized
/// or, if no positive regret exists, a uniform distribution is returned.
fn regret_matching_list(regrets: &Vec<f64>, available_actions: &Vec<usize>) -> Vec<f64> {
    let mut positive_regrets = vec![0.0; regrets.len()];
    for a in 0..regrets.len() {
        if available_actions.contains(&a) && regrets[a] > 0.0 {
            positive_regrets[a] = regrets[a];
        }
    }
    let total_positive: f64 = available_actions.iter().map(|&a| positive_regrets[a]).sum();
    let mut strategy = vec![0.0; regrets.len()];
    if total_positive > 0.0 {
        for &a in available_actions {
            strategy[a] = positive_regrets[a] / total_positive;
        }
    } else {
        let num = available_actions.len();
        for &a in available_actions {
            strategy[a] = 1.0 / num as f64;
        }
    }
    strategy
}

/// The Monte Carlo Counterfactual Regret Minimization (MCCFR) struct.
/// - `G` is a type that implements `AbstractGame<State>`.
/// - `State` is the type representing a game state.
pub struct MCCFR<G, State>
where
    G: AbstractGame<State>,
{
    pub game: G,
    pub num_actions: usize,
    pub num_info_sets: usize,
    /// A vector (indexed by information set) of optional regret vectors.
    pub regrets: Vec<Option<Vec<f64>>>,
    /// A vector (indexed by information set) of optional cumulative strategy vectors.
    pub cumulative_strategy: Vec<Option<Vec<f64>>>,
    /// A set tracking which information sets have been modified.
    pub modified_infosets: HashSet<usize>,
    _state: PhantomData<State>,
}

impl<G, State> MCCFR<G, State>
where
    G: AbstractGame<State>,
    State: Clone,
{
    /// Create a new MCCFR instance.
    pub fn new(game: G, num_info_sets: usize, num_actions: usize) -> Self {
        MCCFR {
            game,
            num_actions,
            num_info_sets,
            regrets: vec![None; num_info_sets],
            cumulative_strategy: vec![None; num_info_sets],
            modified_infosets: HashSet::new(),
            _state: PhantomData,
        }
    }

    /// Ensure that the inner vectors for a given information set index are initialized.
    fn ensure_info_set(&mut self, info_set: usize) {
        if self.regrets[info_set].is_none() {
            self.regrets[info_set] = Some(vec![0.0; self.num_actions]);
        }
        if self.cumulative_strategy[info_set].is_none() {
            self.cumulative_strategy[info_set] = Some(vec![0.0; self.num_actions]);
        }
    }

    /// The recursive external sampling function.
    /// - `state`: the current game state.
    /// - `traverser`: the player index for whom we update regrets.
    /// - `pi`: the probability of reaching this state under the traverser’s strategy.
    /// - `sigma`: the product of sampling probabilities along the trajectory.
    /// - `depth`: the current recursion depth.
    ///
    /// Returns the counterfactual value for the traverser.
    pub fn external_sampling(
        &mut self,
        state: &State,
        traverser: usize,
        pi: f64,
        sigma: f64,
        depth: usize,
    ) -> f64 {
        if depth > 25 {
            println!("Depth exceeded: {}", depth);
            return 0.0;
        }

        if self.game.is_terminal(state) {
            return self.game.get_utility(state)[traverser];
        }

        if self.game.is_chance_node(state) {
            println!("Encountered chance node");
            let (action, prob) = self.game.sample_chance_action(state);
            let next_state = self.game.get_child(state, action);
            return self.external_sampling(&next_state, traverser, pi, sigma * prob, depth + 1);
        }

        let current_player = self.game.get_current_player(state);
        let info_set = self.game.get_information_set(state, current_player);
        self.modified_infosets.insert(info_set);
        self.ensure_info_set(info_set);
        let available_actions = self.game.get_actions(state);

        if current_player == traverser as i32 {
            let strategy =
                regret_matching_list(self.regrets[info_set].as_ref().unwrap(), &available_actions);
            let mut node_value = 0.0;
            let mut action_values = vec![0.0; self.num_actions];

            // Evaluate all available actions.
            for &a in &available_actions {
                let next_state = self.game.apply_action(state, a);
                let action_value = self.external_sampling(
                    &next_state,
                    traverser,
                    pi * strategy[a],
                    sigma,
                    depth + 1,
                );
                action_values[a] = action_value;
                node_value += strategy[a] * action_value;
            }

            // Update regrets.
            if let Some(regret_vec) = self.regrets[info_set].as_mut() {
                for &a in &available_actions {
                    let regret = action_values[a] - node_value;
                    regret_vec[a] += (pi / sigma) * regret;
                }
            }

            // Update cumulative strategy.
            if let Some(cum_strat_vec) = self.cumulative_strategy[info_set].as_mut() {
                for &a in &available_actions {
                    cum_strat_vec[a] += pi * strategy[a];
                }
            }

            node_value
        } else {
            let strategy =
                regret_matching_list(self.regrets[info_set].as_ref().unwrap(), &available_actions);
            // Sample one action according to the strategy.
            let weights: Vec<f64> = available_actions.iter().map(|&a| strategy[a]).collect();
            let dist = WeightedIndex::new(&weights).unwrap();
            let mut rng = thread_rng();
            let chosen_index = dist.sample(&mut rng);
            let chosen_action = available_actions[chosen_index];
            let next_state = self.game.apply_action(state, chosen_action);
            self.external_sampling(&next_state, traverser, pi, sigma * strategy[chosen_action], depth + 1)
        }
    }

    /// Run a single MCCFR iteration for the specified traverser.
    pub fn run_iteration(&mut self, traverser: usize) {
        let initial_state = self.game.get_initial_state();
        self.external_sampling(&initial_state, traverser, 1.0, 1.0, 0);
    }

    /// Compute the average strategy from the cumulative strategy.
    /// Returns a vector (indexed by information set) where each entry is an optional
    /// vector representing the normalized strategy.
    pub fn compute_average_strategy(&self) -> Vec<Option<Vec<f64>>> {
        let mut average_strategy = vec![None; self.num_info_sets];
        for &i in &self.modified_infosets {
            if let Some(ref action_counts) = self.cumulative_strategy[i] {
                let total: f64 = action_counts.iter().sum();
                let avg_strat = if total > 0.0 {
                    action_counts.iter().map(|&c| c / total).collect()
                } else {
                    let num = action_counts.len();
                    vec![1.0 / num as f64; num]
                };
                average_strategy[i] = Some(avg_strat);
            } else {
                average_strategy[i] = Some(vec![]);
            }
        }
        average_strategy
    }
}
