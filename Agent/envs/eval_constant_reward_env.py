import argparse
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# from PPO.ppo_clip import PPOClip
# from ppo_clip.ppo import PPOClip
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from PPO.ppo import PPOClip  
from test_envs_basics import ConstantRewardEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="ConstantRewardEnv")
    p.add_argument("--weights", required=True, help="Path to saved .pt weights")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true", help="Render the environment")
    return p.parse_args()


def make_env(env_name: str, seed: int) -> gym.Env:
    """Create environment - supports both gymnasium environments and custom environments."""
    if env_name == "ConstantRewardEnv":
        env = ConstantRewardEnv()
        env.reset(seed=seed)
        return env
    else:
        try:
            env = gym.make(env_name)
            env.reset(seed=seed)
            return env
        except Exception as e:
            raise ValueError(f"Environment {env_name} not found: {e}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment
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
    
    # Load agent - use discrete flag based on action space
    agent = PPOClip.load(args.weights, obs_dim=obs_dim, act_dim=act_dim, device=device, discrete=is_discrete)

    returns = []
    episode_lengths = []
    solved_count = 0
    SOLVED_THRESHOLD = 1.0  # ConstantRewardEnv always gives reward=1, so threshold is 1.0
    
    print(f"Evaluating agent for {args.episodes} episodes on {args.env}...")
    print(f"Observation space: {obs_dim}D {'(discrete)' if isinstance(env.observation_space, spaces.Discrete) else '(continuous)'}")
    print(f"Action space: {act_dim} {'(discrete)' if is_discrete else 'D (continuous)'}")
    print(f"Solved threshold: {SOLVED_THRESHOLD:.1f} points")
    print("-" * 50)
    
    for ep in range(args.episodes):
        done = False
        ret = 0.0
        steps = 0
        obs, _ = env.reset()
        while not done:
            # Handle discrete observations (convert scalar to 1D tensor)
            if isinstance(env.observation_space, spaces.Discrete):
                obs_t = torch.tensor([obs], dtype=torch.float32, device=device)
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                action, _, _ = agent.select_action(obs_t)
            
            # Handle discrete vs continuous actions
            if is_discrete:
                # For discrete actions, action is a scalar (action index)
                action_np = int(action.item()) if isinstance(action, torch.Tensor) else int(action)
            else:
                # For continuous actions, convert to numpy array
                action_np = action.numpy() if isinstance(action, torch.Tensor) else action
            
            obs, reward, terminated, truncated, _ = env.step(action_np)
            ret += float(reward)
            done = bool(terminated) or bool(truncated)
            steps += 1
        
        is_solved = ret >= SOLVED_THRESHOLD
        if is_solved:
            solved_count += 1
        
        returns.append(ret)
        episode_lengths.append(steps)
        status = " SOLVED" if is_solved else ""
        print(f"Episode {ep+1}: return={ret:.2f}, steps={steps} {status}")

    env.close()
    
    solve_rate = (solved_count / args.episodes * 100) if args.episodes > 0 else 0.0
    
    print("\n" + "="*50)
    print(f"Evaluation Results ({args.episodes} episodes):")
    print(f"  Mean Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"  Mean Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"  Min Return: {np.min(returns):.2f}")
    print(f"  Max Return: {np.max(returns):.2f}")
    print(f"\nSolved Status (threshold: {SOLVED_THRESHOLD:.1f} points):")
    print(f"  Solved episodes: {solved_count} / {args.episodes} ({solve_rate:.1f}%)")
    if solved_count > 0:
        solved_returns = [r for r in returns if r >= SOLVED_THRESHOLD]
        print(f"  Average solved return: {np.mean(solved_returns):.2f}")
    print("="*50)


if __name__ == "__main__":
    main()

