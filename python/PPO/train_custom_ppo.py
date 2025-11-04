import argparse
import os
import time
from typing import Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from rl_bridge.envs.unity_lane_env import UnityLaneEnv
from PPO.ppo_clip import PPOClip, PPOConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--frame_skip", type=int, default=1)      # frame_skip es cuantas veces se ejecuta el step del env
    p.add_argument("--timesteps", type=int, default=200_000)        # uso notacion _ en los numeros por legibilidad
    p.add_argument("--logdir", default="runs/custom_ppo_clip")
    p.add_argument("--save_path", default="runs/custom_ppo_clip/final.pt")
    p.add_argument("--checkpoint_freq", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def make_env(host: str, port: int, frame_skip: int) -> UnityLaneEnv:
    return UnityLaneEnv(host=host, port=port, frame_skip=frame_skip)


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(args.logdir)

    env = make_env(args.host, args.port, args.frame_skip)
    obs, _ = env.reset()
    obs_dim = int(obs.shape[0])
    act_dim = 2  # turn, throttle in [-1,1]

    cfg = PPOConfig()
    agent = PPOClip(obs_dim=obs_dim, act_dim=act_dim, config=cfg, device=device)

    episode_return = 0.0
    episode_len = 0

    total_steps = 0
    best_return = -1e9
    while total_steps < args.timesteps:
        # Collect rollout of length cfg.n_steps
        for _ in range(cfg.n_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, logp, value = agent.select_action(obs_tensor)
            action_np = action.numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action_np)

            agent.buffer.add(obs, action_np, logp, float(reward), bool(terminated), float(value))

            episode_return += float(reward)
            episode_len += 1
            total_steps += 1
            obs = next_obs

            if terminated or truncated:
                writer.add_scalar("rollout/episode_return", episode_return, total_steps)
                writer.add_scalar("rollout/episode_len", episode_len, total_steps)
                obs, _ = env.reset()
                episode_return = 0.0
                episode_len = 0

        # Bootstrap value for GAE
        with torch.no_grad():
            last_value = agent.value(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).item()   # unsqueeze(0) agrega una dimensión extra al principio del tensor, convirtiendo la observación individual en un batch de tamaño 1 --> observacion x.shape    (3,) --> x.unsqueeze(0).shape   (1,3)
        batch = agent.buffer.get(last_value=last_value, gamma=cfg.gamma, lam=cfg.gae_lambda, adv_norm=True)

        # Gradient updates
        metrics = agent.update(batch)
        for k, v in metrics.items():
            writer.add_scalar(f"train/{k}", v, total_steps)

        # Periodic checkpoint
        if args.checkpoint_freq > 0 and total_steps // args.checkpoint_freq != (total_steps - cfg.n_steps) // args.checkpoint_freq:
            ckpt_path = os.path.join(args.logdir, f"ckpt_{total_steps}.pt")
            agent.save(ckpt_path)
            writer.add_text("checkpoint", f"Saved {ckpt_path}", total_steps)

        # Track best episodic return in this window
        if episode_return > best_return:
            best_return = episode_return
            agent.save(os.path.join(args.logdir, "best.pt"))

    env.close()
    writer.close()
    # Save final weights
    agent.save(args.save_path)


if __name__ == "__main__":
    main()


