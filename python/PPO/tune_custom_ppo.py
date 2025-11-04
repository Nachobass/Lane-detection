import argparse
import os
import time
from typing import Dict, Any

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch

from torch.utils.tensorboard import SummaryWriter

from rl_bridge.envs.unity_lane_env import UnityLaneEnv
from PPO.ppo_clip import PPOClip, PPOConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--frame_skip", type=int, default=1)
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logdir", default="runs/custom_ppo_clip/tuning")
    p.add_argument("--storage", default=None)
    p.add_argument("--study_name", default="ppo_clip_custom")
    return p.parse_args()


def suggest(trial: optuna.Trial) -> PPOConfig:
    return PPOConfig(
        gamma=trial.suggest_float("gamma", 0.95, 0.9999, log=True),
        gae_lambda=trial.suggest_float("gae_lambda", 0.8, 0.98),
        clip_range=trial.suggest_float("clip_range", 0.1, 0.35),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        ent_coef=trial.suggest_float("ent_coef", 0.0, 0.02),
        vf_coef=trial.suggest_float("vf_coef", 0.3, 1.0),
        max_grad_norm=trial.suggest_float("max_grad_norm", 0.3, 1.0),
        n_epochs=trial.suggest_categorical("n_epochs", [5, 10, 15]),
        batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
        n_steps=trial.suggest_categorical("n_steps", [512, 1024, 2048]),
        hidden_sizes=trial.suggest_categorical("hidden_sizes", [(128,128), (256,256)]),
    )


def evaluate(env: UnityLaneEnv, agent: PPOClip, episodes: int = 5) -> float:
    device = agent.device
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        steps = 0
        while not done and steps < 5000:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, _, _ = agent.select_action(obs_t)
            obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            ret += float(reward)
            done = bool(terminated) or bool(truncated)
            steps += 1
        returns.append(ret)
    return float(np.mean(returns))


def objective(trial: optuna.Trial, args) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logdir = os.path.join(args.logdir, f"trial_{trial.number}")
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    env = UnityLaneEnv(host=args.host, port=args.port, frame_skip=args.frame_skip)
    obs, _ = env.reset()
    cfg = suggest(trial)
    agent = PPOClip(obs_dim=int(obs.shape[0]), act_dim=2, config=cfg, device=device)

    total = 0
    ep_ret = 0.0
    ep_len = 0
    while total < args.timesteps:
        for _ in range(cfg.n_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, logp, val = agent.select_action(obs_t)
            next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            agent.buffer.add(obs, action.cpu().numpy(), logp, float(reward), bool(terminated), float(val))
            ep_ret += float(reward)
            ep_len += 1
            total += 1
            obs = next_obs
            if terminated or truncated:
                writer.add_scalar("rollout/episode_return", ep_ret, total)
                writer.add_scalar("rollout/episode_len", ep_len, total)
                obs, _ = env.reset()
                ep_ret = 0.0
                ep_len = 0

        with torch.no_grad():
            last_v = agent.value(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).item()
        batch = agent.buffer.get(last_value=last_v, gamma=cfg.gamma, lam=cfg.gae_lambda)
        metrics = agent.update(batch)
        for k, v in metrics.items():
            writer.add_scalar(f"train/{k}", v, total)

    mean_ret = evaluate(env, agent, episodes=5)
    env.close()
    writer.close()
    return mean_ret


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=min(5, max(1, args.trials // 4)))
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )
    study.optimize(lambda t: objective(t, args), n_trials=args.trials, gc_after_trial=True)
    print("Best trial:", study.best_trial.number)
    print("Value:", study.best_value)
    for k, v in study.best_trial.params.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()


