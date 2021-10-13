from agents.networks import PolicyNetwork, MLP
import torch
from agents.buffer import Buffer
import torch.nn.functional as F
import torch.optim as optim


class PPOAgent:
    def __init__(self, env, obs_dim, act_dim, sample_size, gamma, lam,  policy_lr, vf_lr, target_kl=0.01, clip_param=0.2, train_policy_iters=80, train_vf_iters=80, eval_mode=False):
        self.obs_dim, self.act_dim, self.sample_size, self.gamma, self.lam = obs_dim, act_dim, sample_size, gamma, lam
        self.policy = PolicyNetwork()
        self.eval_mode = eval_mode
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.buffer = Buffer(self.obs_dim, self.act_dim, self.sample_size, self.device, self.gamma, self.lam)
        self.policy_lr = policy_lr
        self.vf_lr = vf_lr
        self.train_vf_iters = train_vf_iters
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.vf = MLP(self.obs_dim, 1, activation=torch.tanh).to(self.device)
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=self.vf_lr)
        self.train_policy_iters = train_policy_iters
        self.clip_param = clip_param
        self.steps = 0
        self.env = env
        self.target_kl = target_kl

    def compute_vf_loss(self, obs, ret, v_old):
        # Prediction V(s)
        v = self.vf(obs).squeeze(1)

        # Value loss
        clip_v = v_old + torch.clamp(v - v_old, -self.clip_param, self.clip_param)
        vf_loss = torch.max(F.mse_loss(v, ret), F.mse_loss(clip_v, ret)).mean()
        return vf_loss

    def compute_policy_loss(self, obs, act, adv, log_pi_old):
        # Prediction logπ(s)
        _, log_pi = self.policy(obs, act, use_sample=False)

        # Policy loss
        ratio = torch.exp(log_pi - log_pi_old)
        clip_adv = torch.clamp(ratio, 1. - self.clip_param, 1. + self.clip_param) * adv
        policy_loss = -torch.min(ratio * adv, clip_adv).mean()

        # A sample estimate for KL-divergence, easy to compute
        approx_kl = (log_pi_old - log_pi).mean()
        return policy_loss, approx_kl

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
                v = self.vf(torch.Tensor(obs).to(self.device))
                self.buffer.add(obs, action, reward, done, v)

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
        adv = batch['adv']

        # Prediction logπ_old(s), V_old(s)
        _, log_pi_old = self.policy(obs, act, use_sample=False)
        log_pi_old = log_pi_old.detach()
        v_old = self.vf(obs).squeeze(1)
        v_old = v_old.detach()

        for i in range(self.train_policy_iters):
            policy_loss, kl = self.compute_policy_loss(obs, act, adv, log_pi_old)

            # Early stopping at step i due to reaching max kl
            if kl > 1.5 * self.target_kl:
                break

            # Update policy network parameter
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        # Train value with multiple steps of gradient descent
        for i in range(self.train_vf_iters):
            vf_loss = self.compute_vf_loss(obs, ret, v_old)

            # Update value network parameter
            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()

        # A sample estimate for KL-divergence, easy to compute
        # approx_kl = (log_pi_old - log_pi).mean()

        # Save losses
        # self.policy_losses.append(policy_loss.item())
        # self.vf_losses.append(vf_loss.item())
        # self.kls.append(approx_kl.item())

