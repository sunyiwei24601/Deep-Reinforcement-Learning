from agents.networks import PolicyNetwork
import torch
from agents.buffer import Buffer
import torch.nn.functional as F
import torch.optim as optim


class VPGAgent:
    def __init__(self, env, obs_dim, act_dim, sample_size, gamma, lam,  policy_lr, eval_mode=False):
        self.obs_dim, self.act_dim, self.sample_size, self.gamma, self.lam = obs_dim, act_dim, sample_size, gamma, lam
        self.policy = PolicyNetwork()
        self.eval_mode = eval_mode
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.buffer = Buffer(self.obs_dim, self.act_dim, self.sample_size, self.device, self.gamma, self.lam)
        self.policy_lr = policy_lr
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.steps = 0
        self.env = env

    def run(self):
        step_number = 0
        total_reward = 0.
        obs = self.env.reset()
        done = False
        self.steps = 0
        while not done:
            self.env.render()

            if self.eval_mode:
                action, log_po = self.policy(torch.Tensor(obs).to(self.device))
                action = action.detach().cpu().numpy()
                next_obs, reward, done, _ = self.env.step(action)
            else:
                self.steps += 1

                # Collect experience (s, a, r, s') using some policy
                action, log_pi = self.policy(torch.Tensor(obs).to(self.device))
                action = action.detach().cpu().numpy()
                next_obs, reward, done, _ = self.env.step(action)

                # Add experience to buffer and calculate value function Value
                # v = self.vf(torch.Tensor(obs).to(self.device))
                self.buffer.add(obs, action, reward, done, 0)

                # Start training when the number of experience is equal to sample size
                if self.steps == self.sample_size or done:
                    self.buffer.finish_path()
                    self.train_model()
                    return step_number, total_reward
            total_reward += reward
            step_number += 1
            obs = next_obs
        return step_number, total_reward

    def train_model(self):
        batch = self.buffer.get()
        obs = batch['obs']
        act = batch['act'].detach()
        ret = batch['ret']

        # Prediction logπ(s)
        _, log_pi_old = self.policy(obs, act, use_sample=False)
        log_pi_old = log_pi_old.detach()
        _,  log_pi = self.policy(obs, act, use_sample=False)

        # Policy loss
        policy_loss = -(log_pi * ret).mean()

        # Update policy network parameter
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # A sample estimate for KL-divergence, easy to compute
        approx_kl = (log_pi_old - log_pi).mean()

        # Save losses
        # self.policy_losses.append(policy_loss.item())
        # self.vf_losses.append(vf_loss.item())
        # self.kls.append(approx_kl.item())

