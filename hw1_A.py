import gym
from agents.vpg import Agent


class OnPolicyBuffer:
    def __init__(self):
        raise NotImplementedError


def main(args):
    env = gym.make(args.env)
    agent = Agent()
    buffer = OnPolicyBuffer()

    for epoch in range(args.epoch):
        #  Collect Trajectories
        env.step()
        # Estimate Returns

        # Improve Policy

        # Log Statistics

        pass

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="AntBulletEnv-v0")
    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--algo", required=True, type=str, help="Name of algorithm. It should be one of [pg, pgb, ppo]")

    args = parser.parse_args()
    main(args)
