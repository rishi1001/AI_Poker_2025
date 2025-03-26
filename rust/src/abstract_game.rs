// /// A trait representing an abstract game. The type parameter `State` represents
// /// the type used to encode a game state.
// pub trait AbstractGame<State> {
//     fn is_terminal(&self, state: &State) -> bool;
//     fn get_utility(&self, state: &State) -> Vec<f64>;
//     fn is_chance_node(&self, state: &State) -> bool;
//     fn sample_chance_action(&self, state: &State) -> (usize, f64);
//     fn get_child(&self, state: &State, action: usize) -> State;
//     fn get_current_player(&self, state: &State) -> usize;
//     fn get_information_set(&self, state: &State, player: usize) -> usize;
//     fn get_actions(&self, state: &State) -> Vec<usize>;
//     fn get_initial_state(&self) -> State;
//     fn apply_action(&self, state: &State, action: usize) -> State;
// }

/// The abstract game trait.
pub trait AbstractGame<State> {
    fn is_terminal(&self, state: &State) -> bool;
    fn get_utility(&self, state: &State) -> Vec<f64>;
    fn is_chance_node(&self, state: &State) -> bool;
    fn sample_chance_action(&self, state: &State) -> (usize, f64);
    fn get_child(&self, state: &State, action: usize) -> State;
    fn get_current_player(&self, state: &State) -> i32;
    fn get_information_set(&self, state: &State, player: i32) -> usize;
    fn get_actions(&self, state: &State) -> Vec<usize>;
    fn get_initial_state(&self) -> State;
    fn apply_action(&self, state: &State, action: usize) -> State;
}