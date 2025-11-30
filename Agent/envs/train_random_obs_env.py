import argparse
import os
from typing import Tuple

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter

# from PPO.ppo_clip import PPOClip, PPOConfig
# from ppo_clip.ppo import PPOClip, PPOConfig
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from PPO.ppo import PPOClip, PPOConfig 
from test_envs_basics import RandomObsBinaryRewardEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="RandomObsBinaryRewardEnv")
    p.add_argument("--timesteps", type=int, default=10_000)
    p.add_argument("--logdir", default="runs/random_obs_binary_reward_ppo_clip")
    p.add_argument("--save_path", default="runs/random_obs_binary_reward_ppo_clip/final.pt")
    p.add_argument("--checkpoint_freq", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def make_env(env_name: str, seed: int) -> gym.Env:
    """Create environment - supports both gymnasium environments and custom environments."""
    if env_name == "RandomObsBinaryRewardEnv":
        env = RandomObsBinaryRewardEnv()
        env.reset(seed=seed)
        return env
    else:
        raise ValueError(f"Environment {env_name} not found")


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(args.logdir)
    
    # Print TensorBoard information
    logdir_abs = os.path.abspath(args.logdir)
    # Use relative path for the command (works from project root)
    # If logdir is relative and doesn't already include envs/, prepend it
    logdir_rel = args.logdir
    if not os.path.isabs(logdir_rel) and not logdir_rel.startswith("envs/"):
        logdir_rel = f"envs/{logdir_rel}"
    print("\n" + "="*70)
    print("TensorBoard Logging")
    print("="*70)
    print(f"Log directory (absolute): {logdir_abs}")
    print(f"\nTo view training plots, run:")
    print(f"  tensorboard --logdir {logdir_rel}")
    print(f"\nThen open your browser and navigate to:")
    print(f"  http://localhost:6006")
    print(f"\nOr use the full path:")
    print(f"  http://localhost:6006/#scalars")
    print("="*70 + "\n")

    env = make_env(args.env, args.seed)
    obs, _ = env.reset(seed=args.seed)
    
    # Handle both discrete and continuous observation/action spaces
    if isinstance(env.observation_space, spaces.Discrete):
        obs_dim = 1  # For discrete observations, we'll use a 1D embedding
    else:
        obs_dim = int(obs.shape[0])
    
    if isinstance(env.action_space, spaces.Discrete):
        act_dim = int(env.action_space.n)  # Number of discrete actions
        is_discrete = True
    else:
        act_dim = int(env.action_space.shape[0])  # Continuous action space
        is_discrete = False
    
    # Configure PPO for RandomObsBinaryRewardEnv (simple environment, smaller network is fine)
    cfg = PPOConfig(
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        learning_rate=3e-4,
        ent_coef=0.01,  # Entropy bonus for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        n_epochs=10,
        batch_size=64,
        n_steps=256,  # Smaller rollout for simple environment
        hidden_sizes=(64, 64),  # Smaller network for simple environment
    )

    # Use discrete=True for discrete action spaces
    agent = PPOClip(obs_dim=obs_dim, act_dim=act_dim, config=cfg, device=device, discrete=is_discrete)

    episode_return = 0.0
    episode_len = 0

    total_steps = 0
    best_return = -1e9
    episode_count = 0
    
    # Track recent episode returns for statistics
    recent_returns = []
    recent_returns_window = 10  # Track last 10 episodes
    
    # Track solved status
    # RandomObsBinaryRewardEnv gives reward = obs (which is -1 or 1)
    # We consider it solved if the agent gets positive reward (1.0)
    SOLVED_THRESHOLD = 1.0
    solved_episodes = []
    first_solved_at = None
    first_solved_episode = None
    
    print(f"Training PPO on {args.env}")
    print(f"Observation space: {obs_dim}D {'(discrete)' if isinstance(env.observation_space, spaces.Discrete) else '(continuous)'}")
    print(f"Action space: {act_dim} {'(discrete)' if is_discrete else 'D (continuous)'}")
    print(f"Device: {device}")
    print(f"Total timesteps: {args.timesteps}")
    print(f"\nSolved Criteria: Return >= {SOLVED_THRESHOLD:.1f} points")
    print("  - RandomObsBinaryRewardEnv: Reward = observation (-1 or 1), terminates immediately")
    print("  - Agent should learn to get positive rewards (obs=1)")
    print("-" * 50)
    print(f"{'Episode':<8} {'Return':<10} {'Length':<8} {'Avg Return':<12} {'Best':<10} {'Steps':<10} {'Status':<10}")
    print("-" * 80)
    
    while total_steps < args.timesteps:
        # Collect rollout of length cfg.n_steps
        for _ in range(cfg.n_steps):
            # Handle discrete observations (convert scalar to 1D tensor)
            if isinstance(env.observation_space, spaces.Discrete):
                obs_tensor = torch.tensor([obs], dtype=torch.float32, device=device)
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                action, logp, value = agent.select_action(obs_tensor)
            
            # Handle discrete vs continuous actions
            if is_discrete:
                # For discrete actions, action is a scalar (action index)
                action_np = int(action.item()) if isinstance(action, torch.Tensor) else int(action)
                # Buffer expects array for discrete actions (shape: (act_dim,))
                action_for_buffer = np.array([action_np], dtype=np.float32)
            else:
                # For continuous actions, action is already a numpy-compatible tensor
                action_np = action.numpy() if isinstance(action, torch.Tensor) else action
                action_for_buffer = action_np if isinstance(action_np, np.ndarray) else np.array(action_np, dtype=np.float32)
            
            next_obs, reward, terminated, truncated, _ = env.step(action_np)

            # Convert observation to array for buffer (handle discrete observations)
            obs_for_buffer = np.array([obs], dtype=np.float32) if isinstance(env.observation_space, spaces.Discrete) else obs
            
            agent.buffer.add(obs_for_buffer, action_for_buffer, logp, float(reward), bool(terminated), float(value))

            # Log per-step metrics to TensorBoard
            writer.add_scalar("step/reward", float(reward), total_steps)
            writer.add_scalar("step/value_estimate", float(value), total_steps)
            writer.add_scalar("step/log_prob", float(logp), total_steps)
            
            # Log action statistics (for discrete actions, log which action was taken)
            if is_discrete:
                writer.add_scalar("step/action_taken", action_np, total_steps)
            else:
                # For continuous actions, log action magnitude
                action_magnitude = np.linalg.norm(action_np) if isinstance(action_np, np.ndarray) else abs(action_np)
                writer.add_scalar("step/action_magnitude", action_magnitude, total_steps)

            episode_return += float(reward)
            episode_len += 1
            total_steps += 1
            obs = next_obs

            if terminated or truncated:
                episode_count += 1
                writer.add_scalar("rollout/episode_return", episode_return, total_steps)
                writer.add_scalar("rollout/episode_len", episode_len, total_steps)
                writer.add_scalar("rollout/episode_count", episode_count, total_steps)
                
                # Plot reward vs episode (using episode_count as x-axis)
                writer.add_scalar("episode/reward_vs_episode", episode_return, episode_count)
                writer.add_scalar("episode/length_vs_episode", episode_len, episode_count)
                
                # Track recent returns
                recent_returns.append(episode_return)
                if len(recent_returns) > recent_returns_window:
                    recent_returns.pop(0)
                avg_return = np.mean(recent_returns) if recent_returns else 0.0
                std_return = np.std(recent_returns) if len(recent_returns) > 1 else 0.0
                
                # Log episode-level statistics to TensorBoard
                writer.add_scalar("episode/avg_return_recent", avg_return, total_steps)
                writer.add_scalar("episode/std_return_recent", std_return, total_steps)
                writer.add_scalar("episode/best_return", best_return, total_steps)
                
                # Also log vs episode number for better visualization
                writer.add_scalar("episode/avg_return_recent_vs_episode", avg_return, episode_count)
                writer.add_scalar("episode/best_return_vs_episode", best_return, episode_count)
                
                # Track best return
                is_best = False
                if episode_return > best_return:
                    best_return = episode_return
                    agent.save(os.path.join(args.logdir, "best.pt"))
                    is_best = True
                    writer.add_scalar("episode/new_best_return", best_return, total_steps)
                
                # Check if solved (reward >= threshold)
                is_solved = episode_return >= SOLVED_THRESHOLD
                if is_solved:
                    solved_episodes.append(episode_count)
                    if first_solved_at is None:
                        first_solved_at = total_steps
                        first_solved_episode = episode_count
                        # Save the first solved model
                        agent.save(os.path.join(args.logdir, "solved.pt"))
                        print("\n" + "="*80)
                        print(" ENVIRONMENT SOLVED! ")
                        print("="*80)
                        print(f"First solved at Episode {episode_count} (Step {total_steps})")
                        print(f"Return: {episode_return:.2f} (threshold: {SOLVED_THRESHOLD:.1f})")
                        print(f"Episode Length: {episode_len}")
                        print(f"Model saved to: {os.path.join(args.logdir, 'solved.pt')}")
                        print("="*80 + "\n")
                
                # Log solved status to tensorboard
                writer.add_scalar("rollout/is_solved", 1.0 if is_solved else 0.0, total_steps)
                writer.add_scalar("episode/is_solved_vs_episode", 1.0 if is_solved else 0.0, episode_count)
                
                # Print episode information
                status = "SOLVED!" if is_solved else ("BEST!" if is_best else "")
                print(f"{episode_count:<8} {episode_return:<10.2f} {episode_len:<8} {avg_return:<12.2f} {best_return:<10.2f} {total_steps:<10} {status:<10}")
                
                obs, _ = env.reset()
                episode_return = 0.0
                episode_len = 0

        # Bootstrap value for GAE
        with torch.no_grad():
            # Handle discrete observations
            if isinstance(env.observation_space, spaces.Discrete):
                obs_tensor = torch.tensor([obs], dtype=torch.float32, device=device)
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            last_value = agent.value(obs_tensor.unsqueeze(0)).item()
        
        # Log rollout statistics before getting batch (buffer gets reset after get())
        if agent.buffer.is_full():
            rollout_size = agent.buffer.buffer_size
            writer.add_scalar("train/rollout_avg_reward", np.mean(agent.buffer.rews), total_steps)
            writer.add_scalar("train/rollout_avg_value", np.mean(agent.buffer.vals), total_steps)
            writer.add_scalar("train/rollout_sum_reward", np.sum(agent.buffer.rews), total_steps)
            
            # Log histograms for better visualization (less frequently to avoid clutter)
            if total_steps % (cfg.n_steps * 10) == 0:
                writer.add_histogram("rollout/rewards", agent.buffer.rews, total_steps)
                writer.add_histogram("rollout/values", agent.buffer.vals, total_steps)
                writer.add_histogram("rollout/log_probs", agent.buffer.logp, total_steps)
                if is_discrete:
                    writer.add_histogram("rollout/actions", agent.buffer.acts, total_steps)
        
        batch = agent.buffer.get(last_value=last_value, gamma=cfg.gamma, lam=cfg.gae_lambda, adv_norm=True)

        # Gradient updates
        metrics = agent.update(batch)
        for k, v in metrics.items():
            writer.add_scalar(f"train/{k}", v, total_steps)
        
        # Log learning progress metrics
        if recent_returns:
            writer.add_scalar("progress/avg_return_last_10", np.mean(recent_returns), total_steps)
            writer.add_scalar("progress/episodes_per_rollout", episode_count / max(1, total_steps // cfg.n_steps), total_steps)
        
        # Log batch statistics after update (less frequently)
        if total_steps % (cfg.n_steps * 5) == 0:
            writer.add_histogram("batch/advantages", batch.advantages, total_steps)
            writer.add_histogram("batch/returns", batch.returns, total_steps)
            writer.add_scalar("batch/avg_advantage", batch.advantages.mean().item(), total_steps)
            writer.add_scalar("batch/avg_return", batch.returns.mean().item(), total_steps)
        
        # Print training metrics periodically (every rollout)
        if total_steps % (cfg.n_steps * 5) == 0 or total_steps < cfg.n_steps * 2:  # More frequent at start
            print(f"\n[Training Update @ Step {total_steps}]")
            print(f"  Policy Loss: {metrics.get('loss_policy', 0):.6f}")
            print(f"  Value Loss: {metrics.get('loss_value', 0):.6f}")
            print(f"  Entropy: {metrics.get('entropy', 0):.6f}")
            print(f"  Approx KL: {metrics.get('approx_kl', 0):.6f}")
            print(f"  Clip Fraction: {metrics.get('clip_fraction', 0):.4f}")
            if recent_returns:
                print(f"  Recent Avg Return: {np.mean(recent_returns):.2f}")
            print()

        # Periodic checkpoint and logging
        if args.checkpoint_freq > 0 and total_steps // args.checkpoint_freq != (total_steps - cfg.n_steps) // args.checkpoint_freq:
            ckpt_path = os.path.join(args.logdir, f"ckpt_{total_steps}.pt")
            agent.save(ckpt_path)
            writer.add_text("checkpoint", f"Saved {ckpt_path}", total_steps)
            writer.add_scalar("checkpoint/checkpoint_step", total_steps, total_steps)
            print(f"\n[Checkpoint] Saved checkpoint at step {total_steps} -> {ckpt_path}")
            print(f"  Progress: {100 * total_steps / args.timesteps:.1f}% | Episodes: {episode_count} | Best return: {best_return:.2f}\n")

        # Log hyperparameters to TensorBoard (once at the start)
        if total_steps == cfg.n_steps:
            writer.add_hparams(
                {
                    "gamma": cfg.gamma,
                    "gae_lambda": cfg.gae_lambda,
                    "clip_range": cfg.clip_range,
                    "learning_rate": cfg.learning_rate,
                    "ent_coef": cfg.ent_coef,
                    "vf_coef": cfg.vf_coef,
                    "n_epochs": cfg.n_epochs,
                    "batch_size": cfg.batch_size,
                    "n_steps": cfg.n_steps,
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                    "is_discrete": is_discrete,
                },
                {"initial_best_return": best_return}
            )
        
        # Periodic progress summary
        if total_steps % 50000 == 0 and total_steps > 0:
            print("\n" + "="*70)
            print(f"Progress Summary @ Step {total_steps}/{args.timesteps} ({100 * total_steps / args.timesteps:.1f}%)")
            print("="*70)
            print(f"Total Episodes: {episode_count}")
            print(f"Best Return: {best_return:.2f}")
            if recent_returns:
                print(f"Average Return (last {len(recent_returns)} episodes): {np.mean(recent_returns):.2f}")
                print(f"Std Return (last {len(recent_returns)} episodes): {np.std(recent_returns):.2f}")
            
            # Solved statistics
            if solved_episodes:
                solve_rate = len(solved_episodes) / episode_count * 100
                print(f"\nSolved Status:")
                print(f"  First solved: Episode {first_solved_episode} @ Step {first_solved_at}")
                print(f"  Solved episodes: {len(solved_episodes)} / {episode_count} ({solve_rate:.1f}%)")
                if len(solved_episodes) >= 10:
                    recent_solved = [ep for ep in solved_episodes if ep > episode_count - 10]
                    print(f"  Recent solve rate: {len(recent_solved)}/10 ({len(recent_solved)/10*100:.1f}%)")
            else:
                print(f"\nSolved Status: Not yet solved (need {SOLVED_THRESHOLD:.1f}+ points)")
            
            print("="*70 + "\n")

    env.close()
    writer.close()
    # Save final weights
    agent.save(args.save_path)
    
    print("\n" + "="*70)
    print("Training Completed!")
    print("="*70)
    print(f"Final model saved to: {args.save_path}")
    print(f"\nFinal Statistics:")
    print(f"  Total Episodes: {episode_count}")
    print(f"  Total Steps: {total_steps}")
    print(f"  Best Return: {best_return:.2f}")
    if recent_returns:
        print(f"  Average Return (last {len(recent_returns)} episodes): {np.mean(recent_returns):.2f}")
        print(f"  Std Return (last {len(recent_returns)} episodes): {np.std(recent_returns):.2f}")
    
    # Final solved statistics
    print(f"\nSolved Statistics:")
    if solved_episodes:
        solve_rate = len(solved_episodes) / episode_count * 100
        print(f"   SOLVED! First solved at Episode {first_solved_episode} (Step {first_solved_at})")
        print(f"  Solved episodes: {len(solved_episodes)} / {episode_count} ({solve_rate:.1f}%)")
        if len(solved_episodes) >= 10:
            recent_solved = [ep for ep in solved_episodes[-10:]]
            print(f"  Last 10 episodes solve rate: {len(recent_solved)}/10 ({len(recent_solved)/10*100:.1f}%)")
        print(f"  Solved model saved to: {os.path.join(args.logdir, 'solved.pt')}")
    else:
        print(f"   Not solved (need {SOLVED_THRESHOLD:.1f}+ points)")
        print(f"  Best return achieved: {best_return:.2f} / {SOLVED_THRESHOLD:.1f}")
    
    print("="*70)

    print("\n" + "="*70)
    print("TensorBoard Logging")
    print("="*70)
    print(f"Log directory: {logdir_abs}")
    print(f"\nTo view training plots, run:")
    print(f"  tensorboard --logdir {logdir_rel}")
    print(f"\nThen open your browser and navigate to:")
    print(f"  http://localhost:6006")
    print(f"\nOr use the full path:")
    print(f"  http://localhost:6006/#scalars")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

