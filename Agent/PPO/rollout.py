from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class TrajectoryBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    """Fixed-size on-policy buffer with GAE(lambda)."""

    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.logp = np.zeros(buffer_size, dtype=np.float32)
        self.rews = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.vals = np.zeros(buffer_size, dtype=np.float32)
        self.ptr = 0
        self.full = False
        self.device = device

    def add(self, obs: np.ndarray, act: np.ndarray, logp: float, rew: float, done: bool, val: float):
        i = self.ptr
        self.obs[i] = obs
        self.acts[i] = act
        self.logp[i] = logp
        self.rews[i] = rew
        self.dones[i] = float(done)
        self.vals[i] = val
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
            self.ptr = 0

    def is_full(self) -> bool:
        return self.full

    def compute_gae(self, last_value: float, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
        adv = np.zeros_like(self.rews)
        last_adv = 0.0
        for t in reversed(range(self.buffer_size)):
            nonterminal = 1.0 - self.dones[t]
            next_val = last_value if t == self.buffer_size - 1 else self.vals[t + 1]
            delta = self.rews[t] + gamma * next_val * nonterminal - self.vals[t]
            last_adv = delta + gamma * lam * nonterminal * last_adv
            adv[t] = last_adv
        ret = adv + self.vals
        return adv, ret

    def get(self, last_value: float, gamma: float, lam: float, adv_norm: bool = True) -> TrajectoryBatch:
        advantages, returns = self.compute_gae(last_value, gamma, lam)
        if adv_norm:
            std = np.std(advantages) + 1e-8
            advantages = (advantages - np.mean(advantages)) / std
        batch = TrajectoryBatch(
            observations=torch.tensor(self.obs, device=self.device),
            actions=torch.tensor(self.acts, device=self.device),
            log_probs=torch.tensor(self.logp, device=self.device),
            rewards=torch.tensor(self.rews, device=self.device),
            dones=torch.tensor(self.dones, device=self.device),
            values=torch.tensor(self.vals, device=self.device),
            advantages=torch.tensor(advantages, device=self.device),
            returns=torch.tensor(returns, device=self.device),
        )
        self.full = False
        self.ptr = 0
        return batch


