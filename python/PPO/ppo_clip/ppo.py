from dataclasses import dataclass
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

from .rollout import RolloutBuffer, TrajectoryBatch


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int, int] = (128, 128), activation: nn.Module = nn.Tanh()):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), activation]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.mu = MLP(obs_dim, act_dim, hidden)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(obs)
        log_std = self.log_std.expand_as(mu)
        std = torch.exp(log_std)
        return mu, std

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mu, std = self.forward(obs)
        return torch.distributions.Normal(mu, std)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.dist(obs)
        action = d.sample()
        logp = d.log_prob(action).sum(-1)
        return action, logp, d.mean


@dataclass
class PPOConfig:
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    learning_rate: float = 3e-4
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    batch_size: int = 256
    n_steps: int = 1024
    hidden_sizes: Tuple[int, int] = (128, 128)


class PPOClip(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, config: PPOConfig, device: torch.device):
        super().__init__()
        self.cfg = config
        self.device = device

        self.policy = GaussianPolicy(obs_dim, act_dim, hidden=config.hidden_sizes).to(device)
        self.value = MLP(obs_dim, 1, hidden=config.hidden_sizes).to(device)

        self.optim = optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=config.learning_rate)
        self.buffer = RolloutBuffer(config.n_steps, obs_dim, act_dim, device)

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        obs = obs.to(self.device).unsqueeze(0)
        action, logp, _ = self.policy.act(obs)
        value = self.value(obs).squeeze(-1)
        return action.squeeze(0).cpu(), float(logp.item()), float(value.item())

    def compute_losses(self, batch: TrajectoryBatch) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Policy loss (clipped surrogate)
        dist = self.policy.dist(batch.observations)
        new_logp = dist.log_prob(batch.actions).sum(-1)
        ratio = torch.exp(new_logp - batch.log_probs)
        surr1 = ratio * batch.advantages
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range) * batch.advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Entropy bonus
        entropy = dist.entropy().sum(-1).mean()

        # Value loss (MSE)
        values = self.value(batch.observations).squeeze(-1)
        value_loss = nn.functional.mse_loss(values, batch.returns)

        loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy
        info = {
            "loss_policy": policy_loss.item(),
            "loss_value": value_loss.item(),
            "entropy": entropy.item(),
            "approx_kl": (batch.log_probs - new_logp).mean().abs().item(),
            "clip_fraction": (torch.gt(ratio, 1+self.cfg.clip_range) | torch.lt(ratio, 1-self.cfg.clip_range)).float().mean().item(),
        }
        return loss, info

    def update(self, batch: TrajectoryBatch) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        num_samples = batch.observations.shape[0]
        idx = torch.randperm(num_samples, device=self.device)
        obs = batch.observations[idx]
        acts = batch.actions[idx]
        logp = batch.log_probs[idx]
        adv = batch.advantages[idx]
        ret = batch.returns[idx]

        for epoch in range(self.cfg.n_epochs):
            for start in range(0, num_samples, self.cfg.batch_size):
                end = start + self.cfg.batch_size
                b = TrajectoryBatch(
                    observations=obs[start:end],
                    actions=acts[start:end],
                    log_probs=logp[start:end],
                    rewards=None,  # not used in losses
                    dones=None,    # not used in losses
                    values=None,   # not used in losses
                    advantages=adv[start:end],
                    returns=ret[start:end],
                )
                loss, info = self.compute_losses(b)
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value.parameters()), self.cfg.max_grad_norm)
                self.optim.step()
                for k, v in info.items():
                    metrics[k] = metrics.get(k, 0.0) + float(v)

        # average over number of mini-batches processed
        num_minibatches = self.cfg.n_epochs * max(1, (num_samples + self.cfg.batch_size - 1) // self.cfg.batch_size)
        for k in list(metrics.keys()):
            metrics[k] /= num_minibatches
        return metrics

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_state": self.policy.state_dict(),
            "value_state": self.value.state_dict(),
            "config": self.cfg.__dict__,
        }, path)

    @staticmethod
    def load(path: str, obs_dim: int, act_dim: int, device: torch.device) -> "PPOClip":
        ckpt = torch.load(path, map_location=device)
        cfg = PPOConfig(**ckpt["config"])  # type: ignore[arg-type]
        agent = PPOClip(obs_dim=obs_dim, act_dim=act_dim, config=cfg, device=device)
        agent.policy.load_state_dict(ckpt["policy_state"])  # type: ignore[index]
        agent.value.load_state_dict(ckpt["value_state"])    # type: ignore[index]
        agent.to(device)
        return agent


