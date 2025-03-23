import logging
import gym
from agents.agent import Agent
from gym_env import PokerEnv

class OnePlayerPokerEnv(gym.Env):
    """
    The internal agent is player 1. You are player 0
    """
    def __init__(self, logger: logging.Logger, bot: Agent):
        self.env = PokerEnv()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.logger = logger
        self.bot = bot

    def reset(self, *, seed = None, options = None):
        (obs0, obs1), info = self.env.reset(seed=seed, options=options)
        hand_number = 0
        if options is not None:
            hand_number = options.get("hand_number", 0)
        reward = (0, 0)
        terminated = False
        truncated = False
        while obs0["acting_agent"] == 1 and not terminated:
            info["hand_number"] = hand_number
            bot_action = self.bot.act(obs1, reward, terminated, truncated, info)
            (obs0, obs1), reward, terminated, truncated, info = self.env.step(bot_action)


        return obs0, reward[0], terminated, truncated, info
    
    def step(self, action, hand_number):
        (obs0, obs1), reward, terminated, truncated, info = self.env.step(action)
        while obs0["acting_agent"] == 1 and not terminated:
            info["hand_number"] = hand_number
            bot_action = self.bot.act(obs1, reward, terminated, truncated, info)
            (obs0, obs1), reward, terminated, truncated, info = self.env.step(bot_action)
        return obs0, reward[0], terminated, truncated, info