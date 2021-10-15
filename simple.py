import gym
import pybullet_envs
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Normal
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
from agents.vpg import VPGAgent
from agents.vpg_baseline import VPGBaseLineAgent
from agents.ppo import PPOAgent

env = gym.make("AntBulletEnv-v0")
# env._max_episode_steps = 5000
env.render()
env.reset()
epoch = 100

agent = VPGAgent(env, env.observation_space.shape[0], env.action_space.shape[0],
                 sample_size=1000, gamma=0.99, lam=0.97, policy_lr=1e-3)
agent = VPGBaseLineAgent(env, env.observation_space.shape[0], env.action_space.shape[0],
                 sample_size=1000, gamma=0.99, lam=0.97, policy_lr=1e-3, vf_lr=1e-3)
agent = PPOAgent(env, env.observation_space.shape[0], env.action_space.shape[0],
                 sample_size=1000, gamma=0.99, lam=0.97, policy_lr=1e-3, vf_lr=1e-3)
for e in range(epoch):
    agent.eval_mode = False
    agent.run()
    agent.eval_mode = True
    eval_num_episodes = 0
    eval_sum_returns = 0
    for _ in range(10):
        # Run one episode
        eval_step_length, eval_episode_return = agent.run()

        eval_sum_returns += eval_episode_return
        eval_num_episodes += 1

    eval_average_return = eval_sum_returns / eval_num_episodes if eval_num_episodes > 0 else 0.0
    print("eval_average_return:", eval_average_return)



