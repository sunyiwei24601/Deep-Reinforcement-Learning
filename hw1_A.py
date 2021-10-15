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
from torch.utils.tensorboard import SummaryWriter


def main(env, epoch, algo):

    dir_name = "result\{}_{}_{}.log".format(env, algo, epoch)

    writer = SummaryWriter(log_dir=dir_name)
    env = gym.make(env)
    # env._max_episode_steps = 5000
    env.render()
    env.reset()
    if algo == "pg":
        agent = VPGAgent(env, env.observation_space.shape[0], env.action_space.shape[0],
                     sample_size=1000, gamma=0.99, lam=0.97, policy_lr=1e-3)
    elif algo == "pgb":
        agent = VPGBaseLineAgent(env, env.observation_space.shape[0], env.action_space.shape[0],
                     sample_size=1000, gamma=0.99, lam=0.97, policy_lr=1e-3, vf_lr=1e-3)
    elif algo == 'ppo':
        agent = PPOAgent(env, env.observation_space.shape[0], env.action_space.shape[0],
                     sample_size=1000, gamma=0.99, lam=0.97, policy_lr=1e-3, vf_lr=1e-3)

    step_number = 0

    for e in range(epoch):
        agent.eval_mode = False
        step, _ = agent.run()
        step_number += step
        agent.eval_mode = True
        eval_num_episodes = 0
        eval_sum_returns = 0
        for _ in range(10):
            # Run one episode
            eval_step_length, eval_episode_return = agent.run()

            eval_sum_returns += eval_episode_return
            eval_num_episodes += 1

        eval_average_return = eval_sum_returns / eval_num_episodes if eval_num_episodes > 0 else 0.0
        print("epoch: {}, step_number:{},  eval_average_return:{}".format(e, step_number, eval_average_return))

        writer.add_scalar("learning_curve", eval_average_return, step_number)
        writer.flush()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="AntBulletEnv-v0")
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--algo", default="ppo", type=str, help="Name of algorithm. It should be one of [pg, pgb, ppo]")

    args = parser.parse_args()
    main(args.env, args.epoch, args.algo)
