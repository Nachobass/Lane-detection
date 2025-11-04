import argparse
import os
import time
from typing import Tuple

import numpy as np
import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

# from PPO.ppo_clip import PPOClip, PPOConfig
from ppo_clip.ppo import PPOClip, PPOConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--logdir", default="runs/cartpole_ppo_clip")
    p.add_argument("--save_path", default="runs/cartpole_ppo_clip/final.pt")
    p.add_argument("--checkpoint_freq", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(args.logdir)
    
    # Print TensorBoard information
    logdir_abs = os.path.abspath(args.logdir)
    print("\n" + "="*70)
    print("TensorBoard Logging")
    print("="*70)
    print(f"Log directory: {logdir_abs}")
    print(f"\nTo view training plots, run:")
    print(f"  tensorboard --logdir {logdir_abs}")
    print(f"\nThen open your browser and navigate to:")
    print(f"  http://localhost:6006")
    print(f"\nOr use the full path:")
    print(f"  http://localhost:6006/#scalars")
    print("="*70 + "\n")

    env = make_env(args.env, args.seed)
    obs, _ = env.reset(seed=args.seed)
    obs_dim = int(obs.shape[0])
    act_dim = env.action_space.n  # Discrete action space
    
    # Configure PPO for CartPole (CartPole typically needs different hyperparameters)
    cfg = PPOConfig(
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        learning_rate=3e-4,
        ent_coef=0.01,  # Small entropy bonus for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        n_epochs=4,
        batch_size=64,
        n_steps=2048,
        hidden_sizes=(64, 64),  # Smaller network for CartPole
    )
    agent = PPOClip(obs_dim=obs_dim, act_dim=act_dim, config=cfg, device=device, discrete=True)

    episode_return = 0.0
    episode_len = 0

    total_steps = 0
    best_return = -1e9
    episode_count = 0
    
    # Track recent episode returns for statistics
    recent_returns = []
    recent_returns_window = 10  # Track last 10 episodes
    
    print(f"Training PPO on {args.env}")
    print(f"Observation space: {obs_dim}D")
    print(f"Action space: {act_dim}D (discrete)")
    print(f"Device: {device}")
    print(f"Total timesteps: {args.timesteps}")
    print("-" * 50)
    print(f"{'Episode':<8} {'Return':<10} {'Length':<8} {'Avg Return':<12} {'Best':<10} {'Steps':<10}")
    print("-" * 70)
    
    while total_steps < args.timesteps:
        # Collect rollout of length cfg.n_steps
        for _ in range(cfg.n_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, logp, value = agent.select_action(obs_tensor)
            action_int = int(action.item() if isinstance(action, torch.Tensor) else action)  # Convert to int for discrete action
            next_obs, reward, terminated, truncated, _ = env.step(action_int)

            # Store action as numpy array for buffer (single scalar for discrete actions)
            agent.buffer.add(obs, np.array([float(action_int)], dtype=np.float32), logp, float(reward), bool(terminated), float(value))

            episode_return += float(reward)
            episode_len += 1
            total_steps += 1
            obs = next_obs

            if terminated or truncated:
                episode_count += 1
                writer.add_scalar("rollout/episode_return", episode_return, total_steps)
                writer.add_scalar("rollout/episode_len", episode_len, total_steps)
                writer.add_scalar("rollout/episode_count", episode_count, total_steps)
                
                # Track recent returns
                recent_returns.append(episode_return)
                if len(recent_returns) > recent_returns_window:
                    recent_returns.pop(0)
                avg_return = np.mean(recent_returns) if recent_returns else 0.0
                
                # Track best return
                is_best = False
                if episode_return > best_return:
                    best_return = episode_return
                    agent.save(os.path.join(args.logdir, "best.pt"))
                    is_best = True
                
                # Print episode information
                best_marker = " (BEST!)" if is_best else ""
                print(f"{episode_count:<8} {episode_return:<10.2f} {episode_len:<8} {avg_return:<12.2f} {best_return:<10.2f} {total_steps:<10}{best_marker}")
                
                obs, _ = env.reset()
                episode_return = 0.0
                episode_len = 0

        # Bootstrap value for GAE
        with torch.no_grad():
            last_value = agent.value(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).item()
        batch = agent.buffer.get(last_value=last_value, gamma=cfg.gamma, lam=cfg.gae_lambda, adv_norm=True)

        # Gradient updates
        metrics = agent.update(batch)
        for k, v in metrics.items():
            writer.add_scalar(f"train/{k}", v, total_steps)
        
        # Print training metrics periodically
        if total_steps % (cfg.n_steps * 5) == 0 or total_steps < cfg.n_steps * 2:
            print(f"\n[Training Update @ Step {total_steps}]")
            print(f"  Policy Loss: {metrics.get('loss_policy', 0):.6f}")
            print(f"  Value Loss: {metrics.get('loss_value', 0):.6f}")
            print(f"  Entropy: {metrics.get('entropy', 0):.6f}")
            print(f"  Approx KL: {metrics.get('approx_kl', 0):.6f}")
            print(f"  Clip Fraction: {metrics.get('clip_fraction', 0):.4f}")
            if recent_returns:
                print(f"  Recent Avg Return: {np.mean(recent_returns):.2f}")
            print()

        # Periodic checkpoint
        if args.checkpoint_freq > 0 and total_steps // args.checkpoint_freq != (total_steps - cfg.n_steps) // args.checkpoint_freq:
            ckpt_path = os.path.join(args.logdir, f"ckpt_{total_steps}.pt")
            agent.save(ckpt_path)
            writer.add_text("checkpoint", f"Saved {ckpt_path}", total_steps)
            print(f"\n[Checkpoint] Saved checkpoint at step {total_steps} -> {ckpt_path}")
            print(f"  Progress: {100 * total_steps / args.timesteps:.1f}% | Episodes: {episode_count} | Best return: {best_return:.2f}\n")

        # Periodic progress summary
        if total_steps % 25000 == 0 and total_steps > 0:
            print("\n" + "="*70)
            print(f"Progress Summary @ Step {total_steps}/{args.timesteps} ({100 * total_steps / args.timesteps:.1f}%)")
            print("="*70)
            print(f"Total Episodes: {episode_count}")
            print(f"Best Return: {best_return:.2f}")
            if recent_returns:
                print(f"Average Return (last {len(recent_returns)} episodes): {np.mean(recent_returns):.2f}")
                print(f"Std Return (last {len(recent_returns)} episodes): {np.std(recent_returns):.2f}")
            print("="*70 + "\n")

    env.close()
    writer.close()
    # Save final weights
    agent.save(args.save_path)
    print(f"Training completed! Final model saved to {args.save_path}")


if __name__ == "__main__":
    main()

