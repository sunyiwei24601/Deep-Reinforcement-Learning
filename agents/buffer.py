import torch
import numpy as np


class Buffer(object):
    def __init__(self, obs_dim, act_dim, size, device, gamma=0.99, lam=0.97):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.act_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.don_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.v_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.max_size = 0, size
        self.device = device

    def add(self, obs, act, rew, don, v):
        assert self.ptr < self.max_size  # Buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.don_buf[self.ptr] = don
        self.v_buf[self.ptr] = v
        self.ptr += 1

    def finish_path(self):
        previous_v = 0
        running_ret = 0
        running_adv = 0
        for t in reversed(range(self.ptr)):
            # The next two line computes rewards-to-go, to be targets for the value function
            running_ret = self.rew_buf[t] + self.gamma * (1 - self.don_buf[t]) * running_ret
            self.ret_buf[t] = running_ret

            # The next four lines implement GAE-Lambda advantage calculation
            running_del = self.rew_buf[t] + self.gamma * (1 - self.don_buf[t]) * previous_v - self.v_buf[t]
            running_adv = running_del + self.gamma * self.lam * (1 - self.don_buf[t]) * running_adv
            previous_v = self.v_buf[t]
            self.adv_buf[t] = running_adv
        # The next line implement the advantage normalization trick
        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / self.adv_buf.std()

    def get(self):
        # assert self.ptr == self.max_size  # Buffer has to be full before you can get
        ptr = self.ptr
        self.ptr = 0
        return dict(obs=torch.Tensor(self.obs_buf[:ptr]).to(self.device),
                    act=torch.Tensor(self.act_buf[:ptr]).to(self.device),
                    ret=torch.Tensor(self.ret_buf[:ptr]).to(self.device),
                    adv=torch.Tensor(self.adv_buf[:ptr]).to(self.device),
                    v=torch.Tensor(self.v_buf[:ptr]).to(self.device))