class AbstractGame:
    def get_initial_state(self):
        """Return the initial state of the game."""
        raise NotImplementedError

    def is_terminal(self, state):
        """Return True if state is terminal."""
        raise NotImplementedError

    def get_utility(self, state):
        """
        Return a dictionary mapping player IDs to their utility
        at the terminal state.
        """
        raise NotImplementedError

    def is_chance_node(self, state):
        """Return True if the state is a chance node (e.g. a random event)."""
        raise NotImplementedError

    def sample_chance_action(self, state):
        """For chance nodes, sample and return an action (or outcome)."""
        raise NotImplementedError

    def get_current_player(self, state):
        """Return the player (or chance) whose turn it is to act."""
        raise NotImplementedError

    def get_information_set(self, state, player):
        """
        Return a key (e.g. a string) that uniquely represents the player’s information set
        in this state.
        """
        raise NotImplementedError

    def get_actions(self, state):
        """
        Return a list of actions available at the given state.
        This allows the set of actions to vary with state.
        """
        raise NotImplementedError

    def apply_action(self, state, action):
        """Return the new state after the given action is applied."""
        raise NotImplementedError

    def get_players(self):
        """Return a list of players in the game."""
        raise NotImplementedError
