import argparse
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="BipedalWalker-v3")
    p.add_argument("--weights", required=True, help="Path to saved SB3 .zip weights")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true", help="Render the environment")
    p.add_argument("--solved_threshold", type=float, default=270.0)
    return p.parse_args()


def make_env(env_id: str, render_mode: str | None):
    return gym.make(env_id, render_mode=render_mode)


def maybe_prepare_env(args) -> gym.Env:
    render_mode = None
    if args.render:
        try:
            import pygame  # noqa: F401

            render_mode = "human"
        except ImportError:
            print("Warning: pygame is not installed. Rendering disabled.")
            print("Install pygame via `pip install pygame` or `pip install 'gymnasium[box2d]'`.")

    try:
        env = make_env(args.env, render_mode)
        env.reset(seed=args.seed)
        return env
    except Exception as exc:
        error_str = str(exc).lower()
        error_type = str(type(exc).__name__).lower()
        if "box2d" in error_str or "box2d" in error_type:
            print("\n" + "=" * 70)
            print("ERROR: Box2D is required for BipedalWalker but not installed.")
            print("=" * 70)
            print("Install via conda (recommended): conda install -c conda-forge box2d-py")
            print("or pip (after `xcode-select --install` on macOS): pip install box2d-py")
            print("=" * 70 + "\n")
            raise
        if "pygame" in error_str or "dependencynotinstalled" in error_type:
            print("Error: pygame is required for rendering. Continuing without rendering...")
            env = make_env(args.env, None)
            env.reset(seed=args.seed)
            return env
        raise


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = maybe_prepare_env(args)
    model = PPO.load(args.weights, device=device)

    returns: list[float] = []
    episode_lengths: list[int] = []
    solved_count = 0

    print(f"Evaluating SB3 PPO agent for {args.episodes} episodes on {args.env}...")
    print(f"Solved threshold: {args.solved_threshold:.1f} points")
    print("-" * 50)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ret = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ret += float(reward)
            done = bool(terminated) or bool(truncated)
            steps += 1

        returns.append(ret)
        episode_lengths.append(steps)
        is_solved = ret >= args.solved_threshold
        solved_count += int(is_solved)
        status = " SOLVED" if is_solved else ""
        print(f"Episode {ep + 1}: return={ret:.2f}, steps={steps}{status}")

    env.close()

    solve_rate = solved_count / max(1, args.episodes) * 100

    print("\n" + "=" * 50)
    print(f"Evaluation Results ({args.episodes} episodes):")
    print(f"  Mean Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"  Mean Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"  Min Return: {np.min(returns):.2f}")
    print(f"  Max Return: {np.max(returns):.2f}")
    print(f"\nSolved Status (threshold: {args.solved_threshold:.1f} points):")
    print(f"  Solved episodes: {solved_count} / {args.episodes} ({solve_rate:.1f}%)")
    if solved_count > 0:
        solved_returns = [r for r in returns if r >= args.solved_threshold]
        print(f"  Average solved return: {np.mean(solved_returns):.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()

