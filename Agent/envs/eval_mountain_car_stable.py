import argparse
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO

from train_mountain_car_stable import MountainCarRewardWrapper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="MountainCarContinuous-v0")
    p.add_argument("--weights", required=True, help="Path to SB3 .zip weights")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true", help="Render the environment")
    p.add_argument("--shaping_coef", type=float, default=0.1, help="Must match training wrapper (if used)")
    return p.parse_args()


def make_env(env_id: str, render_mode: str | None, shaping_coef: float):
    env = gym.make(env_id, render_mode=render_mode)
    if shaping_coef != 0.0:
        env = MountainCarRewardWrapper(env, shaping_coef)
    return env


def prepare_env(args) -> gym.Env:
    render_mode = None
    if args.render:
        try:
            import pygame  # noqa: F401

            render_mode = "human"
        except ImportError:
            print("Warning: pygame is not installed. Rendering disabled.")
            print("Install via `pip install pygame` or `pip install 'gymnasium[classic-control]'`.")

    try:
        env = make_env(args.env, render_mode, args.shaping_coef)
        env.reset(seed=args.seed)
        return env
    except Exception as exc:
        error_str = str(exc).lower()
        error_type = type(exc).__name__.lower()
        if "pygame" in error_str or "dependencynotinstalled" in error_type:
            print("pygame missing, running without rendering...")
            env = make_env(args.env, None, args.shaping_coef)
            env.reset(seed=args.seed)
            return env
        raise


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = prepare_env(args)
    model = PPO.load(args.weights, device=device)

    returns: list[float] = []
    episode_lengths: list[int] = []
    solved_count = 0

    print(f"Evaluating SB3 PPO agent on {args.env} for {args.episodes} episodes...")
    print("Solved criterion: reach the goal (terminated=True)")
    print("-" * 50)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ret = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ret += float(info.get("raw_reward", reward))
            done = bool(terminated) or bool(truncated)
            steps += 1

        goal_reached = bool(info.get("goal_reached", False))
        if goal_reached:
            solved_count += 1
        returns.append(ret)
        episode_lengths.append(steps)
        status = " SOLVED" if goal_reached else ""
        print(f"Episode {ep + 1}: return={ret:.2f}, steps={steps}{status}")

    env.close()

    solve_rate = solved_count / max(1, args.episodes) * 100

    print("\n" + "=" * 50)
    print(f"Evaluation Results ({args.episodes} episodes):")
    print(f"  Mean Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"  Mean Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"  Min Return: {np.min(returns):.2f}")
    print(f"  Max Return: {np.max(returns):.2f}")
    print("\nSolved Status:")
    print(f"  Solved episodes: {solved_count} / {args.episodes} ({solve_rate:.1f}%)")
    if solved_count > 0:
        solved_returns = [r for r, solved in zip(returns, [ret >= 0 for ret in returns]) if solved]
        print(f"  Average solved return: {np.mean(solved_returns):.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()

