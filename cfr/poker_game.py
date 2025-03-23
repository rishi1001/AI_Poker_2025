import enum

import torch
import numpy as np
import gym_env
from cfr.mccfr import AbstractGame

class PokerGame(AbstractGame):
    def __init__(self):
        pass

    def _run_gym_env(self, state, ret_env=False):
        actions = state["actions"]
        seed = state["seed"]
        env = gym_env.PokerEnv()
        np.random.seed(seed)
        (obs0, obs1), info = env.reset(seed=seed)
        terminated = False
        truncated = False
        reward0 = reward1 = 0
        info = {}
        for action in actions:
            (obs0, obs1), (reward0, reward1), terminated, truncated, info = env.step(action=action)

        if ret_env:
            return (obs0, obs1), (reward0, reward1), terminated, truncated, info, env
        return (obs0, obs1), (reward0, reward1), terminated, truncated, info
    
    def get_initial_state(self):
        """Return the initial state of the game."""
        return {
            "actions": [],
            "seed": torch.randint(0, 1_000_000_000, (1,)).item(),
        }

    def is_terminal(self, state):
        """Return True if state is terminal."""
        (obs0, obs1), (reward0, reward1), terminated, truncated, info = self._run_gym_env(state)
        return terminated

    def get_utility(self, state):
        """
        Return a dictionary mapping player IDs to their utility
        at the terminal state.
        """
        (obs0, obs1), (reward0, reward1), terminated, truncated, info = self._run_gym_env(state)
        return {0: reward0, 1: reward1}

    def is_chance_node(self, state):
        """Return True if the state is a chance node (e.g. a random event)."""
        return False

    def sample_chance_action(self, state):
        """For chance nodes, sample and return an action (or outcome)."""
        raise NotImplementedError

    def get_current_player(self, state):
        """Return the player (or chance) whose turn it is to act."""
        (obs0, obs1), (reward0, reward1), terminated, truncated, info = self._run_gym_env(state)
        return obs0["acting_agent"]

    def get_information_set(self, state, player):
        """
        Return a key (e.g. a string) that uniquely represents the player’s information set
        in this state.
        """
        (obs0, obs1), (reward0, reward1), terminated, truncated, info = self._run_gym_env(state)
        # player = obs0["acting_agent"] if player == 0 else obs1["acting_agent"]
        # hand = obs0["my_cards"] if player == 0 else obs1["my_cards"]
        # hand_str = f"({hand[0]}, {hand[1]})"
        # return f"{player}_{hand_str}_{state["actions"].join('->')}"
        info_set = str(obs0 if player == 0 else obs1)
        return info_set

    def get_actions(self, state):
        """
        Return a list of actions available at the given state.
        This allows the set of actions to vary with state.
        """
        (obs0, obs1), (reward0, reward1), terminated, truncated, info = self._run_gym_env(state)
        legal_actions = obs0["valid_actions"] if obs0["acting_agent"] == 0 else obs1["valid_actions"]
        tuple_actions = []
        if legal_actions[gym_env.PokerEnv.ActionType.FOLD.value] == 1:
            tuple_actions.append((gym_env.PokerEnv.ActionType.FOLD.value, 0, -1))
        if legal_actions[gym_env.PokerEnv.ActionType.CHECK.value] == 1:
            tuple_actions.append((gym_env.PokerEnv.ActionType.CHECK.value, 0, -1))
        if legal_actions[gym_env.PokerEnv.ActionType.CALL.value] == 1:
            tuple_actions.append((gym_env.PokerEnv.ActionType.CALL.value, 0, -1))
        if legal_actions[gym_env.PokerEnv.ActionType.DISCARD.value] == 1:
            tuple_actions.append((gym_env.PokerEnv.ActionType.DISCARD.value, 0, 0))
            tuple_actions.append((gym_env.PokerEnv.ActionType.DISCARD.value, 0, 1))
        if legal_actions[gym_env.PokerEnv.ActionType.RAISE.value] == 1:
            min_raise = obs0["min_raise"] if obs0["acting_agent"] == 0 else obs1["min_raise"]
            max_raise = obs0["max_raise"] if obs0["acting_agent"] == 0 else obs1["max_raise"]
            pot = obs0["my_bet"] + obs1["my_bet"]
            tuple_actions.append((gym_env.PokerEnv.ActionType.RAISE.value, min_raise, -1))
            tuple_actions.append((gym_env.PokerEnv.ActionType.RAISE.value, max_raise, -1))
            if min_raise < pot < max_raise:
                tuple_actions.append((gym_env.PokerEnv.ActionType.RAISE.value, pot, -1))
            if min_raise < pot//2 < max_raise:
                tuple_actions.append((gym_env.PokerEnv.ActionType.RAISE.value, pot//2, -1))

        # print("tuple_actions", tuple_actions)

        return tuple_actions


    def apply_action(self, state, action):
        """Return the new state after the given action is applied."""
        (obs0, obs1), (reward0, reward1), terminated, truncated, info, env = self._run_gym_env(state, True)

        (obs0, obs1), (reward0, reward1), terminated, truncated, info = env.step(action)
        return {"actions": state["actions"] + [action], "seed": state["seed"]}


    def get_players(self):
        """Return a list of players in the game."""
        return [0, 1]