

from gym_env import PokerEnv
from agents.dqn_agent import DQNPokerAgent

if __name__ == "__main__":
    agent = DQNPokerAgent(0)
    small_blind_player = 0
    
    # make a dummy environment 
    env = PokerEnv()
    
    (obs0, obs1), info = env.reset(options={"small_blind_player": small_blind_player})
    reward = 0
    terminated = False
    truncated = False
    info = {}
    breakpoint()
    
    # run the agent
    action = agent.act(obs0, reward, terminated, truncated, info)
    print(action)
    (obs0, obs1), (reward0, reward1), terminated, truncated, info = env.step(action=action["action"])
    # agent.save("dqn_agent.pth")
    # agent.load("dqn_agent.pth")
    # action = agent.act(observation, reward, terminated, truncated, info)
    # print(action)
    # observation, reward, terminated, truncated, info = env.step(action)
    # action = agent.act(observation, reward, terminated, truncated, info)