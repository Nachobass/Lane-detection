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
    p.add_argument("--env", default="MountainCarContinuous-v0")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--logdir", default="runs/mountain_car_ppo_clip")
    p.add_argument("--save_path", default="runs/mountain_car_ppo_clip/final.pt")
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
    act_dim = int(env.action_space.shape[0])  # Continuous action space
    
    # Configure PPO for Mountain Car Continuous
    # cfg = PPOConfig(
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_range=0.2,
    #     learning_rate=3e-4,
    #     ent_coef=0.01,  # Small entropy bonus for exploration
    #     vf_coef=0.5,
    #     max_grad_norm=0.5,
    #     n_epochs=4,
    #     batch_size=64,
    #     n_steps=2048,
    #     hidden_sizes=(64, 64),  # Smaller network for Mountain Car
    # )
    # cfg = PPOConfig(
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_range=0.2,
    #     learning_rate=3e-4,
    #     ent_coef=0.1,          # un poco más alta para incentivar exploración
    #     vf_coef=0.5,
    #     max_grad_norm=0.5,
    #     n_epochs=4,
    #     batch_size=64,
    #     n_steps=512,           # episodios más cortos, más frecuentes
    #     hidden_sizes=(64, 64),
    # )
    cfg = PPOConfig(
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        learning_rate=1e-4,
        ent_coef=0.05,          # un poco más alta para incentivar exploración
        vf_coef=0.5,
        max_grad_norm=0.5,
        n_epochs=4,
        batch_size=64,
        n_steps=512,           # episodios más cortos, más frecuentes
        hidden_sizes=(64, 64),
    )


    # Note: discrete=False for continuous actions
    agent = PPOClip(obs_dim=obs_dim, act_dim=act_dim, config=cfg, device=device, discrete=False)

    episode_return = 0.0
    episode_len = 0

    total_steps = 0
    best_return = -1e9
    episode_count = 0
    
    # Track recent episode returns for statistics
    recent_returns = []
    recent_returns_window = 10  # Track last 10 episodes
    
    # Track solved status (MountainCar is solved when it reaches the goal)
    solved_episodes = []
    first_solved_at = None
    first_solved_episode = None
    
    print(f"Training PPO on {args.env}")
    print(f"Observation space: {obs_dim}D")
    print(f"Action space: {act_dim}D (continuous)")
    print(f"Device: {device}")
    print(f"Total timesteps: {args.timesteps}")
    print(f"\nSolved Criteria: Reach the goal (terminated=True)")
    print("-" * 50)
    print(f"{'Episode':<8} {'Return':<10} {'Length':<8} {'Avg Return':<12} {'Best':<10} {'Steps':<10} {'Status':<10}")
    print("-" * 80)
    
    while total_steps < args.timesteps:
        # # Collect rollout of length cfg.n_steps
        # for _ in range(cfg.n_steps):
        #     obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        #     with torch.no_grad():
        #         action, logp, value = agent.select_action(obs_tensor)
        #     # For continuous actions, convert to numpy array
        #     action_np = action.numpy() if isinstance(action, torch.Tensor) else action
        #     next_obs, reward, terminated, truncated, info = env.step(action_np)

        #     # Store action for buffer (continuous actions)
        #     agent.buffer.add(obs, action_np, logp, float(reward), bool(terminated), float(value))

        #     episode_return += float(reward)
        #     episode_len += 1
        #     total_steps += 1
        #     obs = next_obs

        #     # Check if solved (reached goal - terminated=True means success in MountainCar)
        #     is_solved = bool(terminated) and not bool(truncated)

        #     if terminated or truncated:
        #         episode_count += 1
        #         writer.add_scalar("rollout/episode_return", episode_return, total_steps)
        #         writer.add_scalar("rollout/episode_len", episode_len, total_steps)
        #         writer.add_scalar("rollout/episode_count", episode_count, total_steps)
                
        #         # Track recent returns
        #         recent_returns.append(episode_return)
        #         if len(recent_returns) > recent_returns_window:
        #             recent_returns.pop(0)
        #         avg_return = np.mean(recent_returns) if recent_returns else 0.0
                
        #         # Track best return
        #         is_best = False
        #         if episode_return > best_return:
        #             best_return = episode_return
        #             agent.save(os.path.join(args.logdir, "best.pt"))
        #             is_best = True
                
        #         # Track solved status
        #         if is_solved:
        #             solved_episodes.append(episode_count)
        #             if first_solved_at is None:
        #                 first_solved_at = total_steps
        #                 first_solved_episode = episode_count
        #                 # Save the first solved model
        #                 agent.save(os.path.join(args.logdir, "solved.pt"))
        #                 print("\n" + "="*80)
        #                 print(" ENVIRONMENT SOLVED! ")
        #                 print("="*80)
        #                 print(f"First solved at Episode {episode_count} (Step {total_steps})")
        #                 print(f"Return: {episode_return:.2f}")
        #                 print(f"Episode Length: {episode_len}")
        #                 print(f"Model saved to: {os.path.join(args.logdir, 'solved.pt')}")
        #                 print("="*80 + "\n")
                
        #         # Log solved status to tensorboard
        #         writer.add_scalar("rollout/is_solved", 1.0 if is_solved else 0.0, total_steps)
                
        #         # Print episode information
        #         status = "SOLVED!" if is_solved else ("BEST!" if is_best else "")
        #         print(f"{episode_count:<8} {episode_return:<10.2f} {episode_len:<8} {avg_return:<12.2f} {best_return:<10.2f} {total_steps:<10} {status:<10}")
                
        #         obs, _ = env.reset()
        #         episode_return = 0.0
        #         episode_len = 0
        for step_i in range(cfg.n_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

            # Phase: random exploration for the first N steps to avoid stagnation
            RANDOM_EXPL_STEPS = 2000
            if total_steps < RANDOM_EXPL_STEPS:
                # sample action uniformly from action space
                action_np = env.action_space.sample().astype(np.float32)
                # compute logp/value placeholders (keep shapes compatible)
                with torch.no_grad():
                    # we still want a logp and a value to fill buffer:
                    # get value only, use dummy logp 0
                    value = agent.value(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(-1).item()
                logp = 0.0
            else:
                with torch.no_grad():
                    action, logp, value = agent.select_action(obs_tensor)
                action_np = action.numpy() if isinstance(action, torch.Tensor) else action
                # ensure float32 numpy array and correct shape
                action_np = np.asarray(action_np, dtype=np.float32).reshape(env.action_space.shape)

            # Clip action to action space bounds (important)
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

            next_obs, reward, terminated, truncated, info = env.step(action_np)

            # Simple reward shaping: small reward proportional to x position
            # This provides directional signal toward the goal and helps PPO.
            # Only apply shaping if not at goal (optionally).
            try:
                pos = float(next_obs[0])  # position is the first observation
            except Exception:
                pos = 0.0
            shaped_reward = float(reward) + 0.1 * pos

            # Debug prints for the first few episodes/steps
            if total_steps < 50:
                print(f"STEP {total_steps} | action {action_np.ravel()} | raw_reward {reward:.3f} | shaped {shaped_reward:.3f} | pos {pos:.3f}")

            # Store action for buffer (continuous actions) — use shaped_reward for learning
            agent.buffer.add(obs, action_np, logp, float(shaped_reward), bool(terminated), float(value))

            episode_return += float(shaped_reward)
            episode_len += 1
            total_steps += 1
            obs = next_obs

            # Check if solved (reached goal - terminated=True means success)
            is_solved = bool(terminated) and not bool(truncated)

            if terminated or truncated:
                episode_count += 1
                # Log raw env reward also to see the difference
                writer.add_scalar("rollout/episode_return_shaped", episode_return, total_steps)
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
                
                # Track solved status (use raw env termination to decide)
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
                        print(f"Return (shaped): {episode_return:.2f}")
                        print(f"Episode Length: {episode_len}")
                        print(f"Model saved to: {os.path.join(args.logdir, 'solved.pt')}")
                        print("="*80 + "\n")
                
                # Log solved status to tensorboard
                writer.add_scalar("rollout/is_solved", 1.0 if is_solved else 0.0, total_steps)
                
                # Print episode information
                status = "SOLVED!" if is_solved else ("BEST!" if is_best else "")
                print(f"{episode_count:<8} {episode_return:<10.2f} {episode_len:<8} {avg_return:<12.2f} {best_return:<10.2f} {total_steps:<10} {status:<10}")
                
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
                print(f"\nSolved Status: Not yet solved (need to reach the goal)")
            
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
        print(f"   Not solved (need to reach the goal)")
    
    print("="*70)


if __name__ == "__main__":
    main()

