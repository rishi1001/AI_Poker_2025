use crate::abstract_game::AbstractGame;
use poker::{Card, Eval, Evaluator, Rank, Suit};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::cmp;
// use std::collections::HashMap;
use hashbrown::HashMap;

//
// State representation for the poker game.
//
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PokerState {
    pub seed: u32,
    pub deck: Vec<i32>,
    pub street: i32,
    pub bets: Vec<i32>, // length 2
    pub discarded_cards: [i32; 2],
    pub drawn_cards: [i32; 2],
    pub player_cards: Vec<Vec<i32>>, // two players’ hands (each two cards)
    pub community_cards: Vec<i32>,   // five cards
    pub acting_agent: i32,
    pub small_blind_player: i32,
    pub big_blind_player: i32,
    pub min_raise: i32,
    pub last_street_bet: i32,
    pub terminated: bool,
    pub winner: Option<i32>, // Some(0) or Some(1) for a win, Some(-1) for a tie, None if not terminated.
}

//
// The observation struct used for a single player’s view of the state.
//
#[derive(Clone, Debug)]
pub struct Observation {
    pub street: i32,
    pub acting_agent: i32,
    pub my_cards: Vec<i32>,
    pub community_cards: Vec<i32>, // always length 5
    pub my_bet: i32,
    pub opp_bet: i32,
    pub opp_discarded_card: i32,
    pub opp_drawn_card: i32,
    pub my_discarded_card: i32,
    pub my_drawn_card: i32,
    pub min_raise: i32,
    pub max_raise: i32,
    pub valid_actions: Vec<i32>, // valid actions encoded as integers
}

//
// A wrapped evaluator that calls into a poker hand evaluator.
//
#[derive(Clone)]
pub struct WrappedEval {
    evaluator: Evaluator,
}

impl WrappedEval {
    pub fn new() -> Self {
        WrappedEval {
            evaluator: Evaluator::new(),
        }
    }

    /// Evaluate a hand with an alternative scoring (aces treated as tens)
    pub fn evaluate(&self, hand: &Vec<Card>, board: &Vec<Card>) -> Eval {
        let cards = hand
            .clone()
            .into_iter()
            .chain(board.clone().into_iter())
            .collect::<Vec<Card>>();
        let reg_score = self.evaluator.evaluate(&cards).unwrap();

        let alt_cards = cards
            .iter()
            .map(|c| {
                if c.rank() == Rank::Ace {
                    Card::new(Rank::Ten, c.suit())
                } else {
                    c.clone()
                }
            })
            .collect::<Vec<Card>>();
        let alt_score = self.evaluator.evaluate(&alt_cards).unwrap();

        if alt_score < reg_score {
            alt_score
        } else {
            reg_score
        }
    }
}

//
// Action types and a helper Action struct.
//
#[derive(Clone, Copy)]
pub enum ActionType {
    FOLD = 0,
    RAISE = 1,
    CHECK = 2,
    CALL = 3,
    DISCARD = 4,
    INVALID = 5,
}

#[derive(Clone)]
pub struct Action {
    pub action_type: ActionType,
    pub raise_amount: i32,
    pub card_to_discard: i32,
}

//
// The poker game implementation.
//
pub struct PokerGame {
    pub small_blind_amount: i32,
    pub big_blind_amount: i32,
    pub max_player_bet: i32,
    // pub ranks: String,
    // pub suits: String,
    pub evaluator: WrappedEval,
}

impl PokerGame {
    pub fn new(small_blind_amount: i32, max_player_bet: i32) -> Self {
        PokerGame {
            small_blind_amount,
            big_blind_amount: small_blind_amount * 2,
            max_player_bet,
            // ranks: "23456789A".to_string(),
            // suits: "dhs".to_string(),
            evaluator: WrappedEval::new(),
        }
    }

    /// Convert an integer card to a Card.
    pub fn int_to_card(&self, card_int: i32) -> Card {
        let rank_index = card_int % 9;
        let suit_index = card_int / 9;
        let rank = match rank_index {
            0 => Rank::Two,
            1 => Rank::Three,
            2 => Rank::Four,
            3 => Rank::Five,
            4 => Rank::Six,
            5 => Rank::Seven,
            6 => Rank::Eight,
            7 => Rank::Nine,
            8 => Rank::Ace,
            _ => panic!(),
        };
        let suit = match suit_index {
            0 => Suit::Diamonds,
            1 => Suit::Hearts,
            2 => Suit::Spades,
            _ => panic!(),
        };
        Card::new(rank, suit)
    }

    /// Create and return the initial game state.
    pub fn get_initial_state(&self) -> PokerState {
        let seed = rand::random::<u32>() % 1_000_000_000;
        let mut rng: StdRng = SeedableRng::seed_from_u64(seed as u64);
        let mut deck: Vec<i32> = (0..27).collect();
        use rand::seq::SliceRandom;
        deck.shuffle(&mut rng);
        let small_blind_player = 0;
        let big_blind_player = 1;
        let mut player_cards = vec![vec![], vec![]];
        for i in 0..2 {
            for _ in 0..2 {
                player_cards[i].push(deck.remove(0));
            }
        }
        let mut community_cards = Vec::new();
        for _ in 0..5 {
            community_cards.push(deck.remove(0));
        }
        let mut bets = vec![0, 0];
        bets[small_blind_player as usize] = self.small_blind_amount;
        bets[big_blind_player as usize] = self.big_blind_amount;
        PokerState {
            seed,
            deck,
            street: 0,
            bets,
            discarded_cards: [-1, -1],
            drawn_cards: [-1, -1],
            player_cards,
            community_cards,
            acting_agent: small_blind_player,
            small_blind_player,
            big_blind_player,
            min_raise: self.big_blind_amount,
            last_street_bet: 0,
            terminated: false,
            winner: None,
        }
    }

    pub fn is_terminal(&self, state: &PokerState) -> bool {
        state.terminated
    }

    pub fn get_utility(&self, state: &PokerState) -> HashMap<i32, f64> {
        if !self.is_terminal(state) {
            panic!("Game is not terminated yet.");
        }
        let pot = cmp::min(state.bets[0], state.bets[1]) as f64;
        let mut util = HashMap::new();
        match state.winner {
            Some(0) => {
                util.insert(0, pot);
                util.insert(1, -pot);
            }
            Some(1) => {
                util.insert(0, -pot);
                util.insert(1, pot);
            }
            Some(-1) => {
                util.insert(0, 0.0);
                util.insert(1, 0.0);
            }
            _ => panic!("Invalid terminal state."),
        }
        util
    }

    pub fn is_chance_node(&self, _state: &PokerState) -> bool {
        false
    }

    pub fn sample_chance_action(&self, _state: &PokerState) -> String {
        panic!("This game has no chance nodes.")
    }

    pub fn get_current_player(&self, state: &PokerState) -> i32 {
        state.acting_agent
    }

    /// Build and return an Observation for a given player.
    fn get_observation(&self, state: &PokerState, player: i32) -> Observation {
        let num_cards_to_reveal = if state.street == 0 {
            0
        } else {
            state.street + 2
        };
        let mut community_cards: Vec<i32> = state
            .community_cards
            .iter()
            .cloned()
            .take(num_cards_to_reveal as usize)
            .collect();
        while community_cards.len() < 5 {
            community_cards.push(-1);
        }
        let max_raise = self.max_player_bet - *state.bets.iter().max().unwrap();
        let mut min_raise = state.min_raise;
        if state.min_raise > max_raise {
            min_raise = max_raise;
        }
        Observation {
            street: state.street,
            acting_agent: state.acting_agent,
            my_cards: state.player_cards[player as usize].clone(),
            community_cards,
            my_bet: state.bets[player as usize],
            opp_bet: state.bets[(1 - player) as usize],
            opp_discarded_card: state.discarded_cards[(1 - player) as usize],
            opp_drawn_card: state.drawn_cards[(1 - player) as usize],
            my_discarded_card: state.discarded_cards[player as usize],
            my_drawn_card: state.drawn_cards[player as usize],
            min_raise,
            max_raise,
            valid_actions: self.get_valid_actions(state, player),
        }
    }

    pub fn get_information_set(&self, state: &PokerState, player: i32) -> String {
        let obs = self.get_observation(state, player);
        PokerGame::compute_information_set_reduced(&obs)
    }

    pub fn get_actions(&self, state: &PokerState) -> Vec<String> {
        let acting = state.acting_agent;
        let valid = self.get_valid_actions(state, acting);
        let mut actions = Vec::new();
        if valid[ActionType::FOLD as usize] == 1 {
            actions.push("FOLD".to_string());
        }
        if valid[ActionType::CHECK as usize] == 1 {
            actions.push("CHECK".to_string());
        }
        if valid[ActionType::CALL as usize] == 1 {
            actions.push("CALL".to_string());
        }
        if valid[ActionType::DISCARD as usize] == 1 {
            actions.push("DISCARD_0".to_string());
            actions.push("DISCARD_1".to_string());
        }
        if valid[ActionType::RAISE as usize] == 1 {
            // We could use the observation struct here as well if desired.
            actions.push("RAISE_MIN".to_string());
            actions.push("RAISE_MAX".to_string());
            actions.push("RAISE_POT".to_string());
            actions.push("RAISE_HALF_POT".to_string());
        }
        actions
    }

    pub fn action_str_to_action_tuple(&self, state: &PokerState, action_str: &str) -> Action {
        match action_str {
            "FOLD" => Action {
                action_type: ActionType::FOLD,
                raise_amount: 0,
                card_to_discard: -1,
            },
            "CHECK" => Action {
                action_type: ActionType::CHECK,
                raise_amount: 0,
                card_to_discard: -1,
            },
            "CALL" => Action {
                action_type: ActionType::CALL,
                raise_amount: 0,
                card_to_discard: -1,
            },
            "DISCARD_0" => Action {
                action_type: ActionType::DISCARD,
                raise_amount: 0,
                card_to_discard: 0,
            },
            "DISCARD_1" => Action {
                action_type: ActionType::DISCARD,
                raise_amount: 0,
                card_to_discard: 1,
            },
            "RAISE_MIN" => {
                let max_raise = self.max_player_bet - *state.bets.iter().max().unwrap();
                let min_raise = cmp::min(state.min_raise, max_raise);
                Action {
                    action_type: ActionType::RAISE,
                    raise_amount: min_raise,
                    card_to_discard: -1,
                }
            }
            "RAISE_MAX" => {
                let amount = self.max_player_bet - *state.bets.iter().max().unwrap();
                Action {
                    action_type: ActionType::RAISE,
                    raise_amount: amount,
                    card_to_discard: -1,
                }
            }
            "RAISE_POT" => {
                let max_raise = self.max_player_bet - *state.bets.iter().max().unwrap();
                let min_raise = cmp::min(state.min_raise, max_raise);
                let pot: i32 = state.bets.iter().sum();
                let safe_bet = cmp::max(min_raise, cmp::min(max_raise, pot));
                Action {
                    action_type: ActionType::RAISE,
                    raise_amount: safe_bet,
                    card_to_discard: -1,
                }
            }
            "RAISE_HALF_POT" => {
                let max_raise = self.max_player_bet - *state.bets.iter().max().unwrap();
                let min_raise = cmp::min(state.min_raise, max_raise);
                let pot: i32 = state.bets.iter().sum();
                let half_pot = pot / 2;
                let safe_bet = cmp::max(min_raise, cmp::min(max_raise, half_pot));
                Action {
                    action_type: ActionType::RAISE,
                    raise_amount: safe_bet,
                    card_to_discard: -1,
                }
            }
            _ => panic!("Invalid action_str {}", action_str),
        }
    }

    pub fn get_players(&self) -> Vec<i32> {
        vec![0, 1]
    }

    /// Determine which actions are valid.
    /// The order is: FOLD, RAISE, CHECK, CALL, DISCARD.
    pub fn get_valid_actions(&self, state: &PokerState, player: i32) -> Vec<i32> {
        let mut valid = vec![1, 1, 1, 1, 1];
        let opponent = 1 - player;
        if state.bets[player as usize] < state.bets[opponent as usize] {
            valid[ActionType::CHECK as usize] = 0;
        }
        if state.bets[player as usize] == state.bets[opponent as usize] {
            valid[ActionType::CALL as usize] = 0;
        }
        if state.discarded_cards[player as usize] != -1 {
            valid[ActionType::DISCARD as usize] = 0;
        }
        if state.street > 1 {
            valid[ActionType::DISCARD as usize] = 0;
        }
        if *state.bets.iter().max().unwrap() == self.max_player_bet {
            valid[ActionType::RAISE as usize] = 0;
        }
        valid
    }

    /// Apply the given action (by its string representation) to the state and return the new state.
    pub fn apply_action(&self, state: &PokerState, action_str: &str) -> PokerState {
        let action = self.action_str_to_action_tuple(state, action_str);
        let mut new_state = state.clone();
        if new_state.terminated {
            panic!("Cannot apply action: game is already terminated.");
        }
        let valid = self.get_valid_actions(&new_state, new_state.acting_agent);
        let mut a_type = action.action_type;
        let raise_amount = action.raise_amount;
        let card_to_discard = action.card_to_discard;
        if valid[a_type as usize] == 0 {
            a_type = ActionType::INVALID;
        }
        if let ActionType::RAISE = a_type {
            if !(new_state.min_raise <= raise_amount
                && raise_amount <= (self.max_player_bet - *new_state.bets.iter().max().unwrap()))
            {
                a_type = ActionType::INVALID;
            }
        }
        let current = new_state.acting_agent;
        let opponent = 1 - current;
        let mut new_street = false;
        match a_type {
            ActionType::FOLD | ActionType::INVALID => {
                new_state.winner = Some(opponent);
                new_state.terminated = true;
            }
            ActionType::CALL => {
                new_state.bets[current as usize] = new_state.bets[opponent as usize];
                if !(new_state.street == 0
                    && current == new_state.small_blind_player
                    && new_state.bets[current as usize] == self.big_blind_amount)
                {
                    new_street = true;
                }
            }
            ActionType::CHECK => {
                if current == new_state.big_blind_player {
                    new_street = true;
                }
            }
            ActionType::RAISE => {
                new_state.bets[current as usize] = new_state.bets[opponent as usize] + raise_amount;
                let raise_so_far = new_state.bets[opponent as usize] - new_state.last_street_bet;
                let max_raise = self.max_player_bet - *new_state.bets.iter().max().unwrap();
                let min_raise_no_limit = raise_so_far + raise_amount;
                new_state.min_raise = cmp::min(min_raise_no_limit, max_raise);
            }
            ActionType::DISCARD => {
                if card_to_discard != -1 {
                    new_state.discarded_cards[current as usize] =
                        new_state.player_cards[current as usize][card_to_discard as usize];
                    let drawn = if !new_state.deck.is_empty() {
                        new_state.deck.remove(0)
                    } else {
                        -1
                    };
                    new_state.drawn_cards[current as usize] = drawn;
                    new_state.player_cards[current as usize][card_to_discard as usize] = drawn;
                }
            }
        }
        if new_street {
            new_state.street += 1;
            new_state.min_raise = self.big_blind_amount;
            new_state.last_street_bet = new_state.bets[0];
            new_state.acting_agent = new_state.small_blind_player;
            if new_state.street > 3 {
                let winner = self.get_winner(&mut new_state);
                new_state.terminated = true;
                new_state.winner = Some(winner);
            }
        } else if let ActionType::DISCARD = a_type {
            // Do nothing
        } else {
            new_state.acting_agent = opponent;
        }
        let max_raise = self.max_player_bet - *new_state.bets.iter().max().unwrap();
        new_state.min_raise = cmp::min(new_state.min_raise, max_raise);
        new_state
    }

    /// Determine the winner at showdown.
    pub fn get_winner(&self, state: &mut PokerState) -> i32 {
        let mut board: Vec<Card> = state
            .community_cards
            .iter()
            .cloned()
            .filter(|&c| c != -1)
            .map(|c| self.int_to_card(c))
            .collect();
        while board.len() < 5 {
            board.push(self.int_to_card(state.deck.remove(0)));
        }
        let p0_cards: Vec<Card> = state.player_cards[0]
            .iter()
            .cloned()
            .filter(|&c| c != -1)
            .map(|c| self.int_to_card(c))
            .collect();
        let p1_cards: Vec<Card> = state.player_cards[1]
            .iter()
            .cloned()
            .filter(|&c| c != -1)
            .map(|c| self.int_to_card(c))
            .collect();
        let score0 = self.evaluator.evaluate(&p0_cards, &board);
        let score1 = self.evaluator.evaluate(&p1_cards, &board);
        if score0 == score1 {
            -1
        } else if score1 < score0 {
            1
        } else {
            0
        }
    }

    /// A reduced version of the information set – build a string from key observations.
    pub fn compute_information_set_reduced(obs: &Observation) -> String {
        let acting_agent = obs.acting_agent;
        let my_cards = obs.my_cards.clone();
        let mut flop_cards = obs
            .community_cards
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<i32>>();
        flop_cards.sort();
        let turn_card = obs.community_cards[3];
        let river_card = obs.community_cards[4];
        let mut my_cards_sorted = my_cards.clone();
        my_cards_sorted.sort();
        // let my_cards_sorted_str = my_cards_sorted
        //     .iter()
        //     .map(|c| c.to_string())
        //     .collect::<Vec<String>>()
        //     .join("-");
        let are_my_two_cards_suited = if my_cards.len() >= 2 {
            (my_cards[0] / 9) == (my_cards[1] / 9)
        } else {
            false
        };
        let mut my_card_numbers = my_cards
            .iter()
            .map(|c| (c % 9).to_string())
            .collect::<Vec<String>>();
        my_card_numbers.sort();
        let my_card_numbers_sorted = my_card_numbers.join(",");
        let mut flop_card_numbers = flop_cards
            .iter()
            .map(|c| (c % 9).to_string())
            .collect::<Vec<String>>();
        flop_card_numbers.sort();
        let flop_card_numbers_sorted = flop_card_numbers.join(",");
        let comm_card_numbers = format!(
            "{},{},{}",
            flop_card_numbers_sorted,
            (turn_card % 9),
            (river_card % 9)
        );
        let valid_actions = obs
            .valid_actions
            .iter()
            .map(|a| a.to_string())
            .collect::<Vec<String>>()
            .join(",");

        let mut suits_map = HashMap::new();
        for card in my_cards_sorted
            .iter()
            .chain(flop_cards.iter())
            .chain(&[turn_card, river_card])
        {
            if *card == -1 {
                continue;
            }
            let suit = card % 9;
            if !suits_map.contains_key(&suit) {
                suits_map.insert(suit, 0);
            }
            *suits_map.get_mut(&suit).unwrap() += 1;
        }
        let is_four_flush = *suits_map.values().max().unwrap() >= 4;
        let is_five_flush = *suits_map.values().max().unwrap() >= 5;

        format!(
            "{}_{}_{}_{}_{}_{}_{}",
            acting_agent,
            my_card_numbers_sorted,
            if is_four_flush {"True"} else {"False"},
            if is_five_flush {"True"} else {"False"},
            if are_my_two_cards_suited {"True"} else {"False"},
            comm_card_numbers,
            valid_actions
        )
    }
}

//
// Implement the AbstractGame trait for PokerGame.
//
impl AbstractGame for PokerGame {
    type State = PokerState;
    type Player = i32;

    fn get_initial_state(&self) -> Self::State {
        self.get_initial_state()
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        self.is_terminal(state)
    }

    fn get_utility(&self, state: &Self::State) -> HashMap<Self::Player, f64> {
        self.get_utility(state)
    }

    fn is_chance_node(&self, state: &Self::State) -> bool {
        self.is_chance_node(state)
    }

    fn sample_chance_action(&self, state: &Self::State) -> String {
        self.sample_chance_action(state)
    }

    fn get_current_player(&self, state: &Self::State) -> Self::Player {
        self.get_current_player(state)
    }

    fn get_information_set(&self, state: &Self::State, player: Self::Player) -> String {
        self.get_information_set(state, player)
    }

    fn get_actions(&self, state: &Self::State) -> Vec<String> {
        self.get_actions(state)
    }

    fn apply_action(&self, state: &Self::State, action: &str) -> Self::State {
        self.apply_action(state, action)
    }

    fn get_players(&self) -> Vec<Self::Player> {
        self.get_players()
    }
}
