// use std::collections::HashMap;
use hashbrown::HashMap;

/// A trait representing an abstract game. The associated types let each game choose its own
/// state representation and player type.
pub trait AbstractGame {
    type State: Clone;
    type Player: Copy + Eq + std::hash::Hash;

    /// Return the initial state of the game.
    fn get_initial_state(&self) -> Self::State;
    /// Return true if the state is terminal.
    fn is_terminal(&self, state: &Self::State) -> bool;
    /// Return a mapping from player to their utility (payoff) in the terminal state.
    fn get_utility(&self, state: &Self::State) -> HashMap<Self::Player, f64>;
    /// Return true if the state is a chance node.
    fn is_chance_node(&self, state: &Self::State) -> bool;
    /// For chance nodes, sample and return an action.
    fn sample_chance_action(&self, state: &Self::State) -> String;
    /// Return the player (or chance) whose turn it is to act.
    fn get_current_player(&self, state: &Self::State) -> Self::Player;
    /// Return a (typically string) key representing the player’s information set.
    fn get_information_set(&self, state: &Self::State, player: Self::Player) -> String;
    /// Return a list of available actions at the given state.
    fn get_actions(&self, state: &Self::State) -> Vec<String>;
    /// Apply an action to the state, returning the new state.
    fn apply_action(&self, state: &Self::State, action: &str) -> Self::State;
    /// Return a list of players in the game.
    fn get_players(&self) -> Vec<Self::Player>;
}
