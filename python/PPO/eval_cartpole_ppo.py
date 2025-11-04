import argparse
import torch
import numpy as np
import gymnasium as gym

# from PPO.ppo_clip import PPOClip
from ppo_clip.ppo import PPOClip


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
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
        if "pygame" in str(e) or "DependencyNotInstalled" in str(type(e).__name__):
            print("Error: pygame is required for CartPole rendering.")
            print("Installing pygame or using non-rendering mode...")
            # Try without rendering
            env = gym.make(args.env, render_mode=None)
            obs, _ = env.reset(seed=args.seed)
            print("Continuing without rendering...")
        else:
            raise
    
    obs_dim = int(obs.shape[0])
    act_dim = env.action_space.n
    
    agent = PPOClip.load(args.weights, obs_dim=obs_dim, act_dim=act_dim, device=device, discrete=True)

    returns = []
    episode_lengths = []
    
    print(f"Evaluating agent for {args.episodes} episodes...")
    
    for ep in range(args.episodes):
        done = False
        ret = 0.0
        steps = 0
        obs, _ = env.reset()
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, _, _ = agent.select_action(obs_t)
            action_int = action.item() if isinstance(action, torch.Tensor) else int(action)
            obs, reward, terminated, truncated, _ = env.step(action_int)
            ret += float(reward)
            done = bool(terminated) or bool(truncated)
            steps += 1
        
        returns.append(ret)
        episode_lengths.append(steps)
        print(f"Episode {ep+1}: return={ret:.2f}, steps={steps}")

    env.close()
    
    print("\n" + "="*50)
    print(f"Evaluation Results ({args.episodes} episodes):")
    print(f"  Mean Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"  Mean Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"  Min Return: {np.min(returns):.2f}")
    print(f"  Max Return: {np.max(returns):.2f}")
    print("="*50)


if __name__ == "__main__":
    main()

