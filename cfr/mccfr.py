import random

from cfr.abstract_game import AbstractGame

class MCCFRTrainer:
    def __init__(self, game: AbstractGame):
        self.game = game
        # We now maintain regret and strategy sums per information set,
        # where each info set’s dictionary keys come from the available actions at that node.
        self.regretSum = {}   # Maps infoSet -> {action: cumulative regret}
        self.strategySum = {} # Maps infoSet -> {action: cumulative strategy probability}

    def get_strategy(self, infoSet, available_actions, realization_weight):
        """
        Compute current strategy at the information set using regret-matching,
        and update the cumulative strategy sum.
        """
        # Initialize regrets for this infoSet if unseen.
        if infoSet not in self.regretSum:
            self.regretSum[infoSet] = {a: 0.0 for a in available_actions}
        regrets = self.regretSum[infoSet]
        normalizing_sum = sum(max(regrets[a], 0) for a in available_actions)
        if normalizing_sum > 0:
            strategy = {a: max(regrets[a], 0) / normalizing_sum for a in available_actions}
        else:
            strategy = {a: 1.0 / len(available_actions) for a in available_actions}

        # Update cumulative strategy sum.
        if infoSet not in self.strategySum:
            self.strategySum[infoSet] = {a: 0.0 for a in available_actions}
        else:
            # Make sure all available actions are present.
            for a in available_actions:
                if a not in self.strategySum[infoSet]:
                    self.strategySum[infoSet][a] = 0.0

        for a in available_actions:
            self.strategySum[infoSet][a] += realization_weight * strategy[a]
        return strategy

    def sample_action(self, strategy):
        """Sample an action based on the provided strategy (a dict mapping actions to probabilities)."""
        r = random.random()
        cumulative_probability = 0.0
        for a, prob in strategy.items():
            cumulative_probability += prob
            if r < cumulative_probability:
                return a
        return list(strategy.keys())[-1]  # Fallback in case of rounding issues

    def cfr(self, state, reach_probs, sample_probs):
        """
        Recursively perform outcome sampling MCCFR on the state.
          - `reach_probs`: dict mapping player to probability of reaching state under current strategy.
          - `sample_probs`: dict mapping player to probability under sampling.
        Returns a dictionary of counterfactual values for each player.
        """
        if self.game.is_terminal(state):
            return self.game.get_utility(state)
        
        if self.game.is_chance_node(state):
            # Sample a chance action and continue.
            action = self.game.sample_chance_action(state)
            next_state = self.game.apply_action(state, action)
            return self.cfr(next_state, reach_probs, sample_probs)

        current_player = self.game.get_current_player(state)
        infoSet = self.game.get_information_set(state, current_player)
        available_actions = self.game.get_actions(state)
        strategy = self.get_strategy(infoSet, available_actions, reach_probs[current_player])
        
        # Outcome sampling: sample one action according to the current strategy.
        sampled_action = self.sample_action(strategy)
        new_state = self.game.apply_action(state, sampled_action)

        # Update sampling probability for the current player.
        new_sample_probs = sample_probs.copy()
        new_sample_probs[current_player] *= strategy[sampled_action]

        # Recursively compute counterfactual utilities.
        utilities = self.cfr(new_state, reach_probs, new_sample_probs)
        util_current = utilities[current_player]

        # Update regrets only for the actions available at this state.
        for a in available_actions:
            if a == sampled_action:
                # Importance sampling correction: divide by probability of sampling a.
                action_util = util_current / strategy[a]
            else:
                action_util = 0
            regret = action_util - util_current
            # Initialize regretSum for infoSet if necessary.
            if infoSet not in self.regretSum:
                self.regretSum[infoSet] = {act: 0.0 for act in available_actions}
            self.regretSum[infoSet][a] += (1.0 / sample_probs[current_player]) * regret

        return utilities

    def train(self, iterations: int):
        """
        Run MCCFR for a specified number of iterations.
        Returns the average strategy for each information set.
        """
        players = self.game.get_players()
        for i in range(iterations):
            reach_probs = {p: 1.0 for p in players}
            sample_probs = {p: 1.0 for p in players}
            initial_state = self.game.get_initial_state()
            self.cfr(initial_state, reach_probs, sample_probs)

            if i % 1000 == 0:
                print("Iteration", i)

        # Compute average strategy from cumulative strategy sums.
        average_strategy = {}
        for infoSet, strat_sum in self.strategySum.items():
            total = sum(strat_sum.values())
            if total > 0:
                average_strategy[infoSet] = {a: strat_sum[a] / total for a in strat_sum}
            else:
                # In case no strategy was accumulated, default to uniform over the actions seen.
                n = len(strat_sum)
                average_strategy[infoSet] = {a: 1.0 / n for a in strat_sum}
        return average_strategy
