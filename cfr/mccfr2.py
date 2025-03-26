import random

from cfr.abstract_game import AbstractGame

def regret_matching_list(regrets, available_actions):
    """
    Compute a strategy (list of probabilities) using regret matching.
    
    :param regrets: A list of regret values (one per action).
    :param available_actions: A list of action indices that are available in the current state.
    :return: A list of probabilities (of length equal to len(regrets)) where only available actions are nonzero.
    """
    # Only consider positive regrets for available actions.
    positive_regrets = [regrets[a] if a in available_actions and regrets[a] > 0 else 0.0 for a in range(len(regrets))]
    total_positive = sum(positive_regrets[a] for a in available_actions)
    strategy = [0.0] * len(regrets)
    if total_positive > 0:
        for a in available_actions:
            strategy[a] = positive_regrets[a] / total_positive
    else:
        # Uniform strategy over available actions.
        num = len(available_actions)
        for a in available_actions:
            strategy[a] = 1.0 / num
    return strategy

class MCCFR:
    def __init__(self, game: AbstractGame, num_info_sets: int, num_actions: int):
        """
        Initialize MCCFR with the provided game and preallocated lists.
        
        :param game: An instance of a game that implements the required methods.
        :param num_info_sets: The total number of information sets.
        :param num_actions: The maximum number of actions per information set.
        """
        self.game = game
        self.num_actions = num_actions
        self.num_info_sets = num_info_sets
        # Initialize the outer lists; inner lists start as None.
        self.regrets = [None] * num_info_sets
        self.cumulative_strategy = [None] * num_info_sets
        self.modified_infosets = set()

    def ensure_info_set(self, info_set):
        """
        Ensure that the inner lists for a given information set index are initialized.
        """
        if self.regrets[info_set] is None:
            self.regrets[info_set] = [0.0] * self.num_actions
        if self.cumulative_strategy[info_set] is None:
            self.cumulative_strategy[info_set] = [0.0] * self.num_actions

    def external_sampling(self, state, traverser, pi, sigma, depth=0):
        """
        Recursive function for external sampling MCCFR.
        
        :param state: The current game state.
        :param traverser: The player (index) for whom we are updating regrets.
        :param pi: The probability of reaching this state under the traverser's strategy.
        :param sigma: The product of sampling probabilities along the sampled trajectory.
        :return: The counterfactual value of the state for the traverser.
        """
        if depth > 25:
            print(depth)
            return 0
        # Terminal state: return payoff for traverser.
        if self.game.is_terminal(state):
            return self.game.get_utility(state)[traverser]
        
        # Chance (nature) node: sample an action.
        if self.game.is_chance_node(state):
            print("never")
            action, prob = self.game.sample_chance_action(state)
            next_state = self.game.get_child(state, action)
            return self.external_sampling(next_state, traverser, pi, sigma * prob, depth=depth+1)
        
        # Get current player and the information set.
        current_player = self.game.get_current_player(state)
        # Assume the game returns an integer index for the information set.
        info_set = self.game.get_information_set(state, current_player)
        self.modified_infosets.add(info_set)
        # Ensure that inner lists are initialized.
        self.ensure_info_set(info_set)
        # Get the list of available actions (as integer indices).
        available_actions = self.game.get_actions(state)
        
        # Decision node for the traverser.
        if current_player == traverser:
            strategy = regret_matching_list(self.regrets[info_set], available_actions)
            node_value = 0.0
            action_values = [0.0] * self.num_actions
            # Evaluate all available actions.
            for a in available_actions:
                next_state = self.game.apply_action(state, a)
                action_value = self.external_sampling(next_state, traverser, pi * strategy[a], sigma, depth=depth+1)
                action_values[a] = action_value
                node_value += strategy[a] * action_value
            # Update regrets.
            for a in available_actions:
                regret = action_values[a] - node_value
                self.regrets[info_set][a] += (pi / sigma) * regret
            # Update cumulative strategy.
            for a in available_actions:
                self.cumulative_strategy[info_set][a] += pi * strategy[a]
            return node_value
        else:
            # Opponent node: sample a single action according to the current strategy.
            strategy = regret_matching_list(self.regrets[info_set], available_actions)
            chosen_action = random.choices(available_actions, weights=[strategy[a] for a in available_actions], k=1)[0]
            next_state = self.game.apply_action(state, chosen_action)
            return self.external_sampling(next_state, traverser, pi, sigma * strategy[chosen_action], depth=depth+1)
    
    def run_iteration(self, traverser):
        """
        Run a single iteration of MCCFR for the given traversing player.
        
        :param traverser: The player index for whom the regrets are being updated.
        """
        initial_state = self.game.get_initial_state()
        self.external_sampling(initial_state, traverser, pi=1, sigma=1)

    def compute_average_strategy(self):
        """
        Convert the cumulative strategy (stored as a list of lists) into an average strategy.
        
        :return: A list of lists representing the normalized average strategy.
        """
        average_strategy = [None] * self.num_info_sets
        for i in self.modified_infosets:
            action_counts = self.cumulative_strategy[i]
        # for action_counts in cumulative_strategy:
            if action_counts is None:
                # If no actions were recorded, assume a uniform strategy.
                avg_strat = []
            else:
                total = sum(action_counts)
                if total > 0:
                    avg_strat = [count / total for count in action_counts]
                else:
                    num_actions = len(action_counts)
                    avg_strat = [1.0 / num_actions for _ in range(num_actions)]
            # average_strategy.append(avg_strat)
            average_strategy[i] = avg_strat
        return average_strategy
