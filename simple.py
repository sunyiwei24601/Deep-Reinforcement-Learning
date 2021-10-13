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


def reward_to_go(rewards, gamma):
    n = len(rewards)
    rtgs = np.zeros_like(rewards)
    for i in reversed(range(n)):
        rtgs[i] = rewards[i] + (gamma * rtgs[i+1] if i+1 < n else 0)
    return rtgs


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(28, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 8)
        self.mean_policy = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.ReLU(), self.fc3)

        self.log_std = np.ones(8, dtype=np.float32)
        self.log_std = nn.Parameter(torch.Tensor(self.log_std))

    def forward(self, x):
        mean = self.mean_policy(x)
        std = torch.exp(self.log_std)
        prob = Normal(mean, self.log_std)

        action = prob.sample()
        logp = prob.log_prob(action).sum(dim=-1)
        #         print(mean.mean(), variance.mean())
        return action, logp

policy = PolicyNet()
learning_rate = 3e-4
batch_size = 1000
gamma = 0.99
optimizer = torch.optim.RMSprop(policy.parameters(), lr=learning_rate)
# Batch History
state_pool = []
action_pool = []
reward_pool = []
prob_pool = []
epoch = 1000


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



