use poker::{Card, Eval, Evaluator, Rank, Suit};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng, SeedableRng};
use std::cmp;
use std::collections::HashMap;

use crate::abstract_game::AbstractGame;

/// A complete representation of a Poker state.
/// (This mirrors the Python dictionary state.)
#[derive(Clone, Debug)]
pub struct PokerState {
    pub seed: u32,
    pub deck: Vec<i32>, // remaining deck (cards 0..26)
    pub street: i32,    // game street
    pub bets: [i32; 2], // bets for each player
    pub discarded_cards: [i32; 2],
    pub drawn_cards: [i32; 2],
    pub player_cards: [[i32; 2]; 2], // two players, each with two cards
    pub community_cards: [i32; 5],
    pub acting_agent: i32,
    pub small_blind_player: i32,
    pub big_blind_player: i32,
    pub min_raise: i32,
    pub last_street_bet: i32,
    pub terminated: bool,
    pub winner: Option<i32>, // Some(0) or Some(1) for a win, Some(-1) for tie; None if not yet decided.
}

/// An observation of the game from a single player’s view.
#[derive(Clone, Debug)]
pub struct Observation {
    pub street: i32,
    pub acting_agent: i32,
    pub my_cards: Vec<i32>,
    pub community_cards: Vec<i32>, // length 5 (with unrevealed cards set to -1)
    pub my_bet: i32,
    pub opp_bet: i32,
    pub opp_discarded_card: i32,
    pub opp_drawn_card: i32,
    pub my_discarded_card: i32,
    pub my_drawn_card: i32,
    pub min_raise: i32,
    pub max_raise: i32,
    pub valid_actions: Vec<i32>, // 5-element vector (order: FOLD, RAISE, CHECK, CALL, DISCARD)
}

/// Action types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActionType {
    Fold = 0,
    Raise = 1,
    Check = 2,
    Call = 3,
    Discard = 4,
    Invalid = 5,
}

/// An action represented as a triple: (action type, raise amount, card_to_discard).
#[derive(Debug, Clone)]
pub struct Action {
    pub action_type: ActionType,
    pub amount: i32,
    pub card_to_discard: i32,
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

/// The concrete Poker game.
pub struct PokerGame {
    pub small_blind_amount: i32,
    pub max_player_bet: i32,
    pub ranks: &'static str,
    pub suits: &'static str,
    pub evaluator: WrappedEval,
}

impl PokerGame {
    pub fn new(small_blind_amount: i32, max_player_bet: i32) -> Self {
        PokerGame {
            small_blind_amount,
            max_player_bet,
            ranks: "23456789A",
            suits: "dhs", // diamonds, hearts, spades
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

    /// Initialize a single–player observation from the global state.
    fn get_single_player_obs(&self, state: &PokerState, player: i32) -> Observation {
        let num_cards_to_reveal = if state.street == 0 {
            0
        } else {
            state.street + 2
        };
        let mut community_cards = state.community_cards.to_vec();
        // Only reveal the first num_cards_to_reveal cards; set the rest to -1.
        for i in num_cards_to_reveal as usize..5 {
            community_cards[i] = -1;
        }
        let opp = 1 - player;
        Observation {
            street: state.street,
            acting_agent: state.acting_agent,
            my_cards: state.player_cards[player as usize].to_vec(),
            community_cards,
            my_bet: state.bets[player as usize],
            opp_bet: state.bets[opp as usize],
            opp_discarded_card: state.discarded_cards[opp as usize],
            opp_drawn_card: state.drawn_cards[opp as usize],
            my_discarded_card: state.discarded_cards[player as usize],
            my_drawn_card: state.drawn_cards[player as usize],
            min_raise: state.min_raise,
            max_raise: self.max_player_bet - cmp::max(state.bets[0], state.bets[1]),
            valid_actions: self._get_valid_actions(state, player),
        }
    }

    /// Compute the valid actions (as a vector of 0/1 values) for a player.
    /// The order is: FOLD, RAISE, CHECK, CALL, DISCARD.
    fn _get_valid_actions(&self, state: &PokerState, player: i32) -> Vec<i32> {
        let mut valid = vec![1, 1, 1, 1, 1];
        let opp = 1 - player;
        if state.bets[player as usize] < state.bets[opp as usize] {
            valid[ActionType::Check as usize] = 0;
        }
        if state.bets[player as usize] == state.bets[opp as usize] {
            valid[ActionType::Call as usize] = 0;
        }
        if state.discarded_cards[player as usize] != -1 {
            valid[ActionType::Discard as usize] = 0;
        }
        if state.street > 1 {
            valid[ActionType::Discard as usize] = 0;
        }
        if cmp::max(state.bets[0], state.bets[1]) == self.max_player_bet {
            valid[ActionType::Raise as usize] = 0;
        }
        valid
    }

    /// Convert an integer action (0..8) into an Action tuple.
    pub fn action_int_to_action_tuple(&self, state: &PokerState, action_int: usize) -> Action {
        match action_int {
            0 => Action {
                action_type: ActionType::Fold,
                amount: 0,
                card_to_discard: -1,
            },
            1 => Action {
                action_type: ActionType::Check,
                amount: 0,
                card_to_discard: -1,
            },
            2 => Action {
                action_type: ActionType::Call,
                amount: 0,
                card_to_discard: -1,
            },
            3 => {
                // Choose the lower card index from the player's hand.
                let current_hand = state.player_cards[state.acting_agent as usize];
                let lower_card_idx = if current_hand[0] % 9 <= current_hand[1] % 9 {
                    0
                } else {
                    1
                };
                Action {
                    action_type: ActionType::Discard,
                    amount: 0,
                    card_to_discard: lower_card_idx as i32,
                }
            }
            4 => {
                // Choose the higher card index from the player's hand.
                let current_hand = state.player_cards[state.acting_agent as usize];
                let higher_card_idx = if current_hand[0] % 9 >= current_hand[1] % 9 {
                    0
                } else {
                    1
                };
                Action {
                    action_type: ActionType::Discard,
                    amount: 0,
                    card_to_discard: higher_card_idx as i32,
                }
            }
            5 => {
                let max_bet = cmp::max(state.bets[0], state.bets[1]);
                let min_raise = cmp::min(state.min_raise, self.max_player_bet - max_bet);
                Action {
                    action_type: ActionType::Raise,
                    amount: min_raise,
                    card_to_discard: -1,
                }
            }
            6 => {
                let max_bet = cmp::max(state.bets[0], state.bets[1]);
                Action {
                    action_type: ActionType::Raise,
                    amount: self.max_player_bet - max_bet,
                    card_to_discard: -1,
                }
            }
            7 => {
                let max_bet = cmp::max(state.bets[0], state.bets[1]);
                let max_raise = self.max_player_bet - max_bet;
                let min_raise = cmp::min(state.min_raise, max_raise);
                let pot: i32 = state.bets.iter().sum();
                let safe_bet = cmp::max(min_raise, cmp::min(max_raise, pot));
                Action {
                    action_type: ActionType::Raise,
                    amount: safe_bet,
                    card_to_discard: -1,
                }
            }
            8 => {
                let max_bet = cmp::max(state.bets[0], state.bets[1]);
                let max_raise = self.max_player_bet - max_bet;
                let min_raise = cmp::min(state.min_raise, max_raise);
                let pot: i32 = state.bets.iter().sum();
                let half_pot = pot / 2;
                let safe_bet = cmp::max(min_raise, cmp::min(max_raise, half_pot));
                Action {
                    action_type: ActionType::Raise,
                    amount: safe_bet,
                    card_to_discard: -1,
                }
            }
            _ => panic!("wtf - invalid action_int {}", action_int),
        }
    }

    /// Determine the winner at showdown.
    /// This function builds the board from revealed community cards (drawing from the deck if necessary)
    /// and then uses the evaluator to compare players’ hands.
    /// Determine the winner at showdown.
    pub fn _get_winner(&self, state: &mut PokerState) -> i32 {
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

    /// Compute an information set index from an observation.
    /// (For simplicity the observation is encoded as a single integer using mixed–radix encoding.)
    pub fn compute_information_set(&self, obs: &Observation) -> usize {
        // Sort the flop (first three community cards).
        let mut flop = obs.community_cards[..3].to_vec();
        flop.sort();
        let turn = obs.community_cards[3];
        let river = obs.community_cards[4];

        let mut suits_map: HashMap<i32, i32> = HashMap::new();
        for &card in obs
            .my_cards
            .iter()
            .chain(flop.iter())
            .chain(std::iter::once(&turn))
            .chain(std::iter::once(&river))
        {
            if card == -1 {
                continue;
            }
            let suit = card / 9;
            *suits_map.entry(suit).or_insert(0) += 1;
        }
        let max_count = suits_map.values().cloned().max().unwrap_or(0);
        let is_four_flush = max_count >= 4;
        let is_five_flush = max_count >= 5;

        let convert_card = |card: i32| -> i32 {
            if card == -1 {
                -1
            } else {
                1 + ((card + 1) % 9)
            }
        };

        let mut my_cards_nums: Vec<i32> = obs.my_cards.iter().map(|&c| convert_card(c)).collect();
        my_cards_nums.sort();
        let mut community_cards_nums: Vec<i32> = obs
            .community_cards
            .iter()
            .map(|&c| convert_card(c))
            .collect();
        community_cards_nums.sort();

        let valid_actions_str: String = obs.valid_actions.iter().map(|v| v.to_string()).collect();

        let valid_actions_map: HashMap<&str, i32> = [
            ("11101", 0),
            ("10010", 1),
            ("11011", 2),
            ("10101", 3),
            ("11010", 4),
            ("11100", 5),
            ("10100", 6),
            ("10011", 7),
        ]
        .iter()
        .cloned()
        .collect();

        let continuation_cost = obs.opp_bet - obs.my_bet;
        let pot = obs.opp_bet + obs.my_bet;
        let pot_odds = if pot != 0 {
            continuation_cost as f64 / pot as f64
        } else {
            0.0
        };
        let binned_pot_odds = (pot_odds * 3.0).floor() as i32;

        let player = obs.acting_agent;
        let my_hand_numbers_int = tuple_to_int_2(&my_cards_nums);
        let are_my_two_cards_suited = if obs.my_cards[0] / 9 == obs.my_cards[1] / 9 {
            1
        } else {
            0
        };
        let flush_number = if !is_four_flush {
            0
        } else if !is_five_flush {
            1
        } else {
            2
        };
        let community_card_numbers_int = tuple_to_int_5(&community_cards_nums);
        let valid_actions_number = *valid_actions_map
            .get(valid_actions_str.as_str())
            .unwrap_or(&0);
        // let fields = vec![
        //     player,
        //     my_hand_numbers_int,
        //     are_my_two_cards_suited,
        //     flush_number,
        //     community_card_numbers_int,
        //     valid_actions_number,
        //     binned_pot_odds,
        // ];
        // let radices = vec![2, 55, 2, 3, 2002, 8, 3];
        // encode_fields(&fields, &radices)

        let my_hand_i32 = obs.my_cards.iter().map(|x| *x).collect::<Vec<_>>();
        let community_cards_i32 = obs
            .community_cards
            .clone()
            .into_iter()
            .filter(|x| *x != -1)
            .collect::<Vec<_>>();
        let opp_drawn_cards_i32 = if obs.opp_drawn_card != -1 {
            vec![obs.opp_drawn_card]
        } else {
            vec![]
        };
        let mut known_cards_i32 = vec![obs.my_discarded_card, obs.opp_discarded_card]
            .into_iter()
            .filter(|x| *x != -1)
            .collect::<Vec<_>>();

        known_cards_i32.extend(my_hand_i32.clone());
        known_cards_i32.extend(community_cards_i32.clone());
        known_cards_i32.extend(opp_drawn_cards_i32.clone());


        let equity = self.monte_carlo_equity(
            &my_hand_i32,
            &community_cards_i32,
            &opp_drawn_cards_i32,
            &known_cards_i32,
        );

        let binned_equity = (equity * 8.0).floor() as i32;

        let fields = vec![
            binned_equity,
            flush_number,
            valid_actions_number as i32,
            binned_pot_odds as i32,
        ];
        let radices = vec![9, 3, 8, 3];

        encode_fields(&fields, &radices)
    }

    fn monte_carlo_equity(
        &self,
        my_cards: &Vec<i32>,
        community_cards: &Vec<i32>,
        opp_drawn_card: &Vec<i32>,
        known_cards: &Vec<i32>, // my_cards + community + opp_discarded + opp_drawn
    ) -> f32 {
        let mut rng = thread_rng();
        let mut wins = 0;
        let num_simulations = 300;
        let total_needed = 7 - community_cards.len() - opp_drawn_card.len();

        // Generate full deck of 27 cards
        let all_cards: Vec<_> = (0..27).collect();
        let non_shown_cards: Vec<_> = all_cards
            .into_iter()
            .filter(|c| !known_cards.contains(c))
            .collect();

        for _ in 0..num_simulations {
            if non_shown_cards.len() < total_needed {
                panic!("monte_carlo error: not enough cards to simulate"); // Not enough cards to simulate
            }

            let mut drawn = non_shown_cards.clone();
            drawn.shuffle(&mut rng);
            let drawn_cards: Vec<_>= drawn.into_iter().take(total_needed).collect();

            let mut opp_cards = opp_drawn_card.to_vec();
            opp_cards.extend_from_slice(&drawn_cards[..2 - opp_drawn_card.len()]);

            let mut full_community = community_cards.to_vec();
            full_community.extend_from_slice(&drawn_cards[2 - opp_drawn_card.len()..]);

            if self.evaluate_hand(my_cards, &opp_cards, &full_community) {
                wins += 1;
            }
        }

        wins as f32 / num_simulations as f32
    }

    fn evaluate_hand(&self, my_cards: &[i32], opp_cards: &[i32], community_cards: &[i32]) -> bool {
        let my_hand: Vec<Card> = my_cards
            .iter()
            .map(|&c| self.int_to_card(c))
            .collect();
        let opp_hand: Vec<Card> = opp_cards
            .iter()
            .map(|&c| self.int_to_card(c))
            .collect();
        let community: Vec<Card> = community_cards
            .iter()
            .map(|&c| self.int_to_card(c))
            .collect();

        let my_rank = self.evaluator.evaluate(&my_hand, &community);
        let opp_rank = self.evaluator.evaluate(&opp_hand, &community);

        my_rank < opp_rank
    }
}

impl AbstractGame<PokerState> for PokerGame {
    fn is_terminal(&self, state: &PokerState) -> bool {
        state.terminated
    }

    fn get_utility(&self, state: &PokerState) -> Vec<f64> {
        if !state.terminated {
            panic!("Game is not terminated yet.");
        }
        let pot = cmp::min(state.bets[0], state.bets[1]) as f64;
        match state.winner {
            Some(winner) => {
                if winner == 0 {
                    vec![pot, -pot]
                } else if winner == 1 {
                    vec![-pot, pot]
                } else {
                    vec![0.0, 0.0]
                }
            }
            None => panic!("Winner not determined in a terminated state."),
        }
    }

    fn is_chance_node(&self, _state: &PokerState) -> bool {
        false
    }

    fn sample_chance_action(&self, _state: &PokerState) -> (usize, f64) {
        unimplemented!("This game has no chance nodes.")
    }

    fn get_child(&self, state: &PokerState, action: usize) -> PokerState {
        self.apply_action(state, action)
    }

    fn get_current_player(&self, state: &PokerState) -> i32 {
        state.acting_agent
    }

    fn get_information_set(&self, state: &PokerState, player: i32) -> usize {
        let obs = self.get_single_player_obs(state, player);
        self.compute_information_set(&obs)
    }

    fn get_actions(&self, state: &PokerState) -> Vec<usize> {
        let acting = state.acting_agent;
        let valid = self._get_valid_actions(state, acting);
        let mut actions = Vec::new();
        if valid[ActionType::Fold as usize] == 1
            && valid[ActionType::Check as usize] == 0
            && valid[ActionType::Discard as usize] == 0
        {
            actions.push(0);
        }
        if valid[ActionType::Check as usize] == 1 {
            actions.push(1);
        }
        if valid[ActionType::Call as usize] == 1 {
            actions.push(2);
        }
        if valid[ActionType::Discard as usize] == 1 {
            actions.push(3);
            actions.push(4);
        }
        if valid[ActionType::Raise as usize] == 1 {
            actions.push(5);
            actions.push(6);
            actions.push(7);
            actions.push(8);
        }
        actions
    }

    fn get_initial_state(&self) -> PokerState {
        let seed = rand::thread_rng().gen_range(0..1_000_000_000);
        let mut deck: Vec<i32> = (0..27).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
        deck.shuffle(&mut rng);

        let small_blind_player = 0;
        let big_blind_player = 1;
        let mut player_cards = [[-1; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                player_cards[i][j] = deck.remove(0);
            }
        }
        let mut community_cards = [-1; 5];
        for i in 0..5 {
            community_cards[i] = deck.remove(0);
        }
        let mut bets = [0, 0];
        bets[small_blind_player] = self.small_blind_amount;
        bets[big_blind_player] = self.small_blind_amount * 2;
        PokerState {
            seed,
            deck,
            street: 0,
            bets,
            discarded_cards: [-1, -1],
            drawn_cards: [-1, -1],
            player_cards,
            community_cards,
            acting_agent: small_blind_player as i32,
            small_blind_player: small_blind_player as i32,
            big_blind_player: big_blind_player as i32,
            min_raise: self.small_blind_amount * 2,
            last_street_bet: 0,
            terminated: false,
            winner: None,
        }
    }

    fn apply_action(&self, state: &PokerState, action_int: usize) -> PokerState {
        let mut new_state = state.clone();
        if new_state.terminated {
            panic!("Cannot apply action: game is already terminated.");
        }
        let action = self.action_int_to_action_tuple(&new_state, action_int);
        let current = new_state.acting_agent;
        let opp = 1 - current;
        let valid = self._get_valid_actions(&new_state, current);
        let mut a_type = action.action_type;
        if valid[a_type as usize] == 0 {
            a_type = ActionType::Invalid;
        }
        if a_type == ActionType::Raise {
            let max_bet = cmp::max(new_state.bets[0], new_state.bets[1]);
            if !(new_state.min_raise <= action.amount
                && action.amount <= (self.max_player_bet - max_bet))
            {
                a_type = ActionType::Invalid;
            }
        }
        let mut winner: Option<i32> = None;
        let mut new_street = false;

        if a_type == ActionType::Fold || a_type == ActionType::Invalid {
            winner = Some(opp);
            new_state.terminated = true;
            new_state.winner = winner;
        } else if a_type == ActionType::Call {
            new_state.bets[current as usize] = new_state.bets[opp as usize];
            if !(new_state.street == 0
                && current == new_state.small_blind_player
                && new_state.bets[current as usize] == self.small_blind_amount * 2)
            {
                new_street = true;
            }
        } else if a_type == ActionType::Check {
            if current == new_state.big_blind_player {
                new_street = true;
            }
        } else if a_type == ActionType::Raise {
            new_state.bets[current as usize] = new_state.bets[opp as usize] + action.amount;
            let raise_so_far = new_state.bets[opp as usize] - new_state.last_street_bet;
            let max_raise = self.max_player_bet - cmp::max(new_state.bets[0], new_state.bets[1]);
            let min_raise_no_limit = raise_so_far + action.amount;
            new_state.min_raise = cmp::min(min_raise_no_limit, max_raise);
        } else if a_type == ActionType::Discard {
            if action.card_to_discard != -1 {
                new_state.discarded_cards[current as usize] =
                    new_state.player_cards[current as usize][action.card_to_discard as usize];
                let drawn = if !new_state.deck.is_empty() {
                    new_state.deck.remove(0)
                } else {
                    -1
                };
                new_state.drawn_cards[current as usize] = drawn;
                new_state.player_cards[current as usize][action.card_to_discard as usize] = drawn;
            }
        }

        if new_street {
            new_state.street += 1;
            new_state.min_raise = self.small_blind_amount * 2;
            new_state.last_street_bet = new_state.bets[0]; // bets should be equal now
            new_state.acting_agent = new_state.small_blind_player;
            if new_state.street > 3 {
                let w = self._get_winner(&mut new_state);
                new_state.terminated = true;
                new_state.winner = Some(w);
            }
        } else if a_type != ActionType::Discard {
            new_state.acting_agent = opp;
        }
        new_state.min_raise = cmp::min(
            new_state.min_raise,
            self.max_player_bet - cmp::max(new_state.bets[0], new_state.bets[1]),
        );
        new_state
    }
}

/// Encode a vector of values into a single integer using mixed–radix encoding.
pub fn encode_fields(values: &Vec<i32>, radices: &Vec<i32>) -> usize {
    values
        .iter()
        .zip(radices.iter())
        .fold(0usize, |acc, (&v, &r)| acc * (r as usize) + (v as usize))
}

/// Decode an integer back into a vector of values using the provided radices.
pub fn decode_fields(mut index: usize, radices: &Vec<i32>) -> Vec<i32> {
    let mut values = vec![0; radices.len()];
    for (i, &radix) in radices.iter().rev().enumerate() {
        values[radices.len() - 1 - i] = (index % (radix as usize)) as i32;
        index /= radix as usize;
    }
    values
}

/// Compute “n choose k”.
pub fn comb(n: i32, k: i32) -> i32 {
    if k > n {
        return 0;
    }
    let mut result = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Map a sorted 5–tuple (represented as a Vec of 5 integers) to an integer in 0..2001.
pub fn tuple_to_int_5(t: &Vec<i32>) -> i32 {
    let mut y = vec![0; 5];
    for i in 0..5 {
        y[i] = t[i] + i as i32;
    }
    let n = 14;
    let k = 5;
    let mut rank = 0;
    let mut prev = 0;
    for i in 0..k {
        for j in prev..y[i] {
            rank += comb(n - j - 1, k as i32 - i as i32 - 1);
        }
        prev = y[i] + 1;
    }
    rank
}

/// Inverse of `tuple_to_int_5`: Map an integer in 0..2001 back to a sorted 5–tuple.
pub fn int_to_tuple_5(mut rank: i32) -> Vec<i32> {
    let n = 14;
    let k = 5;
    let mut y = vec![];
    let mut x_val = 0;
    for _ in 0..k {
        loop {
            let count = comb(n - x_val - 1, k - y.len() as i32 - 1);
            if rank < count {
                y.push(x_val);
                x_val += 1;
                break;
            } else {
                rank -= count;
                x_val += 1;
            }
        }
    }
    y.iter().enumerate().map(|(i, &v)| v - i as i32).collect()
}

/// Map a sorted 2–tuple (as Vec of 2 integers) to an integer in 0..54.
pub fn tuple_to_int_2(t: &Vec<i32>) -> i32 {
    let mut y = vec![0; 2];
    for i in 0..2 {
        y[i] = t[i] + i as i32;
    }
    let n = 11;
    let k = 2;
    let mut rank = 0;
    let mut prev = 0;
    for i in 0..k {
        for j in prev..y[i] {
            rank += comb(n - j - 1, k as i32 - i as i32 - 1);
        }
        prev = y[i] + 1;
    }
    rank
}

/// Inverse of `tuple_to_int_2`: Map an integer in 0..54 back to a sorted 2–tuple.
pub fn int_to_tuple_2(mut rank: i32) -> Vec<i32> {
    let n = 11;
    let k = 2;
    let mut y = vec![];
    let mut x_val = 0;
    for _ in 0..k {
        loop {
            let count = comb(n - x_val - 1, k - y.len() as i32 - 1);
            if rank < count {
                y.push(x_val);
                x_val += 1;
                break;
            } else {
                rank -= count;
                x_val += 1;
            }
        }
    }
    y.iter().enumerate().map(|(i, &v)| v - i as i32).collect()
}

/// Return a pretty–printed string for a list of action probabilities.
pub fn pretty_action_list(action_probabilities: &Vec<f64>) -> String {
    format!(
        "F:{:.3}|Ch:{:.3}|Ca:{:.3}|D0:{:.3}|D1:{:.3}|Rmin:{:.3}|Rmax:{:.3}|Rp:{:.3}|Rhp:{:.3}",
        action_probabilities[0],
        action_probabilities[1],
        action_probabilities[2],
        action_probabilities[3],
        action_probabilities[4],
        action_probabilities[5],
        action_probabilities[6],
        action_probabilities[7],
        action_probabilities[8]
    )
}

// Assume our NiceInfoSet type returned by decode_infoset_int is defined as:
#[derive(Debug)]
pub struct NiceInfoSet {
    pub binned_equity: i32,
    pub flush_number: i32,
    pub valid_actions_number: i32,
    pub binned_pot_odds: i32,
}

pub fn decode_infoset_int(infoset: usize) -> NiceInfoSet {
    let radices = vec![9, 3, 8, 3];
    let x = decode_fields(infoset, &radices);
    NiceInfoSet {
        binned_equity: x[0],
        flush_number: x[1],
        valid_actions_number: x[2],
        binned_pot_odds: x[3],
    }
}
