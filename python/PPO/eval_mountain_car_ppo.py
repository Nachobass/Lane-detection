import argparse
import torch
import numpy as np
import gymnasium as gym

# from PPO.ppo_clip import PPOClip
from ppo_clip.ppo import PPOClip


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="MountainCarContinuous-v0")
    p.add_argument("--weights", required=True, help="Path to saved .pt weights")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true", help="Render the environment")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Handle rendering - only use "human" mode if explicitly requested and pygame is available
    render_mode = None
    if args.render:
        try:
            import pygame
            render_mode = "human"
        except ImportError:
            print("Warning: pygame is not installed. Rendering disabled.")
            print("To enable rendering, install pygame: pip install pygame")
            print("Or install with: pip install 'gymnasium[classic-control]'")
            render_mode = None
    
    try:
        env = gym.make(args.env, render_mode=render_mode)
        obs, _ = env.reset(seed=args.seed)
    except Exception as e:
        error_str = str(e).lower()
        if "pygame" in error_str or "DependencyNotInstalled" in str(type(e).__name__):
            print("Warning: pygame is not installed. Rendering disabled.")
            print("To enable rendering, install pygame: pip install pygame")
            print("Or install with: pip install 'gymnasium[classic-control]'")
            # Try without rendering
            env = gym.make(args.env, render_mode=None)
            obs, _ = env.reset(seed=args.seed)
            print("Continuing without rendering...")
        else:
            raise
    
    obs_dim = int(obs.shape[0])
    act_dim = int(env.action_space.shape[0])  # Continuous action space
    
    # Note: discrete=False for continuous actions
    agent = PPOClip.load(args.weights, obs_dim=obs_dim, act_dim=act_dim, device=device, discrete=False)

    returns = []
    episode_lengths = []
    solved_count = 0
    solved_episode_indices = []
    
    print(f"Evaluating agent for {args.episodes} episodes on {args.env}...")
    print(f"Observation space: {obs_dim}D")
    print(f"Action space: {act_dim}D (continuous)")
    print(f"Solved criteria: Reach the goal (terminated=True)")
    print("-" * 50)
    
    for ep in range(args.episodes):
        done = False
        ret = 0.0
        steps = 0
        obs, _ = env.reset()
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, _, _ = agent.select_action(obs_t)
            # For continuous actions, convert to numpy array
            action_np = action.numpy() if isinstance(action, torch.Tensor) else action
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(action_np)

            ret += float(reward)
            done = bool(terminated) or bool(truncated)
            steps += 1
        
        # Check if solved (reached goal - terminated=True means success)
        is_solved = bool(terminated) and not bool(truncated)
        if is_solved:
            solved_count += 1
            solved_episode_indices.append(ep)
        
        returns.append(ret)
        episode_lengths.append(steps)
        status = " SOLVED" if is_solved else ""
        print(f"Episode {ep+1}: return={ret:.2f}, steps={steps}{status}")

    env.close()
    
    solve_rate = (solved_count / args.episodes * 100) if args.episodes > 0 else 0.0
    
    print("\n" + "="*50)
    print(f"Evaluation Results ({args.episodes} episodes):")
    print(f"  Mean Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"  Mean Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"  Min Return: {np.min(returns):.2f}")
    print(f"  Max Return: {np.max(returns):.2f}")
    print(f"\nSolved Status:")
    print(f"  Solved episodes: {solved_count} / {args.episodes} ({solve_rate:.1f}%)")
    if solved_count > 0:
        solved_returns = [returns[i] for i in solved_episode_indices]
        print(f"  Average solved return: {np.mean(solved_returns):.2f}")
        print(f"  Average solved steps: {np.mean([episode_lengths[i] for i in solved_episode_indices]):.2f}")
    print("="*50)


if __name__ == "__main__":
    main()

