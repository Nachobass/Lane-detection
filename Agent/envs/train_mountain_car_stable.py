import argparse
import os
from collections import deque
from typing import Deque, Optional

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="MountainCarContinuous-v0")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--logdir", default="runs/mountain_car_sb3_ppo")
    p.add_argument("--save_path", default="runs/mountain_car_sb3_ppo/final_model")
    p.add_argument("--checkpoint_freq", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shaping_coef", type=float, default=0.1, help="Reward shaping coefficient * position")
    p.add_argument("--solved_reward", type=float, default=90.0, help="Solved if shaped return >= threshold")
    return p.parse_args()


class MountainCarRewardWrapper(gym.Wrapper):
    """Adds small potential-based shaping to help the agent climb the hill."""

    def __init__(self, env: gym.Env, shaping_coef: float):
        super().__init__(env)
        self.shaping_coef = shaping_coef

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pos = float(obs[0]) if obs is not None else 0.0
        shaped_reward = reward + self.shaping_coef * pos
        info["raw_reward"] = reward
        info["shaped_reward"] = shaped_reward
        info["goal_reached"] = bool(terminated and not truncated)
        return obs, shaped_reward, terminated, truncated, info


def make_env(env_name: str, seed: int, shaping_coef: float):
    def _init():
        env = gym.make(env_name)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        if shaping_coef != 0.0:
            env = MountainCarRewardWrapper(env, shaping_coef)
        return env

    return _init


class EpisodeLoggingCallback(BaseCallback):
    """Mirrors the rich logging/solved tracking from the original PPO loop."""

    def __init__(
        self,
        writer: SummaryWriter,
        logdir: str,
        checkpoint_freq: int,
        total_timesteps: int,
        best_model_path: str,
        solved_model_path: str,
        solved_reward: float,
        env_name: str,
        obs_space,
        action_space,
        device: str,
        summary_interval: int = 25_000,
        recent_returns_window: int = 10,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.writer = writer
        self.logdir = logdir
        self.checkpoint_freq = checkpoint_freq
        self.total_timesteps = total_timesteps
        self.best_model_path = best_model_path
        self.solved_model_path = solved_model_path
        self.solved_reward = solved_reward
        self.env_name = env_name
        self.obs_space = obs_space
        self.action_space = action_space
        self.device = device
        self.summary_interval = summary_interval
        self.recent_returns: Deque[float] = deque(maxlen=recent_returns_window)
        self.recent_lengths: Deque[int] = deque(maxlen=recent_returns_window)
        self.best_return = -np.inf
        self.episode_count = 0
        self.last_checkpoint_step = 0
        self.last_summary_step = 0
        self.solved_episodes: list[int] = []
        self.first_solved_step: Optional[int] = None
        self.first_solved_episode: Optional[int] = None

    def _on_training_start(self) -> None:
        obs_dim = getattr(self.obs_space, "shape", None)
        obs_str = f"{int(np.prod(obs_dim))}D" if obs_dim else str(self.obs_space)
        act_space = self.action_space
        act_str = (
            f"{act_space.shape[0]}D continuous"
            if hasattr(act_space, "shape")
            else str(act_space)
        )
        print(f"\nTraining SB3 PPO on {self.env_name}")
        print(f"Observation space: {obs_str}")
        print(f"Action space: {act_str}")
        print(f"Device: {self.device}")
        print(f"Total timesteps: {self.total_timesteps}")
        print(f"Solved reward threshold: {self.solved_reward:.1f}")
        print("-" * 60)
        print(
            f"{'Episode':<8} {'Return':<10} {'Length':<8} "
            f"{'Avg Return':<12} {'Best':<10} {'Steps':<10} {'Status':<10}"
        )
        print("-" * 80)

    def _save_checkpoint(self) -> None:
        ckpt_path = os.path.join(self.logdir, f"ckpt_{self.num_timesteps}")
        self.model.save(ckpt_path)  # type: ignore[union-attr]
        self.writer.add_text("checkpoint", f"Saved {ckpt_path}", self.num_timesteps)
        progress = 100 * self.num_timesteps / max(1, self.total_timesteps)
        print(f"\n[Checkpoint] Saved at step {self.num_timesteps} ({progress:.1f}% progress)\n")

    def _maybe_print_summary(self) -> None:
        if self.num_timesteps - self.last_summary_step < self.summary_interval or self.num_timesteps == 0:
            return
        self.last_summary_step = self.num_timesteps
        avg_return = float(np.mean(self.recent_returns)) if self.recent_returns else 0.0
        std_return = float(np.std(self.recent_returns)) if len(self.recent_returns) > 1 else 0.0
        progress = 100 * self.num_timesteps / max(1, self.total_timesteps)
        print("\n" + "=" * 70)
        print(f"Progress Summary @ Step {self.num_timesteps}/{self.total_timesteps} ({progress:.1f}%)")
        print("=" * 70)
        print(f"Total Episodes: {self.episode_count}")
        print(f"Best Return: {self.best_return:.2f}")
        if self.recent_returns:
            print(f"Average Return (last {len(self.recent_returns)} episodes): {avg_return:.2f}")
            print(f"Std Return (last {len(self.recent_returns)} episodes): {std_return:.2f}")
        if self.solved_episodes:
            solve_rate = len(self.solved_episodes) / max(1, self.episode_count) * 100
            print(f"Solved episodes: {len(self.solved_episodes)} / {self.episode_count} ({solve_rate:.1f}%)")
            print(f"First solved episode: {self.first_solved_episode} @ step {self.first_solved_step}")
        else:
            print("Solved status: not yet solved")
        print("=" * 70 + "\n")

    def _handle_episode(self, episode_return: float, episode_len: int, goal_reached: bool) -> None:
        self.episode_count += 1
        self.recent_returns.append(episode_return)
        self.recent_lengths.append(episode_len)
        avg_return = float(np.mean(self.recent_returns)) if self.recent_returns else 0.0
        std_return = float(np.std(self.recent_returns)) if len(self.recent_returns) > 1 else 0.0

        self.writer.add_scalar("rollout/episode_return", episode_return, self.num_timesteps)
        self.writer.add_scalar("rollout/episode_len", episode_len, self.num_timesteps)
        self.writer.add_scalar("rollout/episode_count", self.episode_count, self.num_timesteps)
        self.writer.add_scalar("episode/reward_vs_episode", episode_return, self.episode_count)
        self.writer.add_scalar("episode/length_vs_episode", episode_len, self.episode_count)

        best_marker = ""
        if episode_return > self.best_return:
            self.best_return = episode_return
            self.model.save(self.best_model_path)  # type: ignore[union-attr]
            self.writer.add_scalar("episode/new_best_return", self.best_return, self.num_timesteps)
            best_marker = "BEST!"

        is_solved = goal_reached or episode_return >= self.solved_reward
        if is_solved:
            self.solved_episodes.append(self.episode_count)
            if self.first_solved_step is None:
                self.first_solved_step = self.num_timesteps
                self.first_solved_episode = self.episode_count
                self.model.save(self.solved_model_path)  # type: ignore[union-attr]
                print("\n" + "=" * 80)
                print(" ENVIRONMENT SOLVED! ")
                print("=" * 80)
                print(f"First solved at Episode {self.episode_count} (Step {self.num_timesteps})")
                print(f"Return: {episode_return:.2f}")
                print(f"Solved model saved to: {self.solved_model_path}")
                print("=" * 80 + "\n")
            best_marker = "SOLVED!"

        self.writer.add_scalar("episode/avg_return_recent", avg_return, self.num_timesteps)
        self.writer.add_scalar("episode/std_return_recent", std_return, self.num_timesteps)
        self.writer.add_scalar("episode/avg_return_recent_vs_episode", avg_return, self.episode_count)
        self.writer.add_scalar("episode/best_return", self.best_return, self.num_timesteps)
        self.writer.add_scalar("episode/best_return_vs_episode", self.best_return, self.episode_count)
        self.writer.add_scalar("rollout/is_solved", 1.0 if is_solved else 0.0, self.num_timesteps)
        self.writer.add_scalar("episode/is_solved_vs_episode", 1.0 if is_solved else 0.0, self.episode_count)

        print(
            f"{self.episode_count:<8} {episode_return:<10.2f} {episode_len:<8} "
            f"{avg_return:<12.2f} {self.best_return:<10.2f} {self.num_timesteps:<10} {best_marker:<10}"
        )

    def _on_step(self) -> bool:
        if (
            self.checkpoint_freq > 0
            and self.num_timesteps - self.last_checkpoint_step >= self.checkpoint_freq
        ):
            self._save_checkpoint()
            self.last_checkpoint_step = self.num_timesteps

        infos = self.locals.get("infos", [])  # type: ignore[attr-defined]
        for info in infos:
            episode_info = info.get("episode")
            if episode_info is None:
                continue
            goal_reached = bool(info.get("goal_reached", False))
            self._handle_episode(float(episode_info["r"]), int(episode_info["l"]), goal_reached)

        self._maybe_print_summary()
        return True

    def final_stats(self) -> dict:
        avg_return = float(np.mean(self.recent_returns)) if self.recent_returns else 0.0
        std_return = float(np.std(self.recent_returns)) if len(self.recent_returns) > 1 else 0.0
        return {
            "episodes": self.episode_count,
            "best_return": self.best_return,
            "avg_recent_return": avg_return,
            "std_recent_return": std_return,
            "solved": bool(self.solved_episodes),
            "first_solved_episode": self.first_solved_episode,
            "first_solved_step": self.first_solved_step,
            "num_solved": len(self.solved_episodes),
        }


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter(args.logdir)

    logdir_abs = os.path.abspath(args.logdir)
    logdir_rel = args.logdir
    if not os.path.isabs(logdir_rel) and not logdir_rel.startswith("envs/"):
        logdir_rel = f"envs/{logdir_rel}"
    print("\n" + "=" * 70)
    print("TensorBoard Logging")
    print("=" * 70)
    print(f"Log directory (absolute): {logdir_abs}")
    print("\nTo view training plots, run:")
    print(f"  tensorboard --logdir {logdir_rel}")
    print("\nThen open your browser and navigate to:")
    print("  http://localhost:6006")
    print("\nOr use the full path:")
    print("  http://localhost:6006/#scalars")
    print("=" * 70 + "\n")

    env = DummyVecEnv([make_env(args.env, args.seed, args.shaping_coef)])
    env = VecMonitor(env)
    obs_space = env.observation_space
    action_space = env.action_space

    best_model_path = os.path.join(args.logdir, "best_model")
    solved_model_path = os.path.join(args.logdir, "solved_model")
    callback = EpisodeLoggingCallback(
        writer=writer,
        logdir=args.logdir,
        checkpoint_freq=args.checkpoint_freq,
        total_timesteps=args.timesteps,
        best_model_path=best_model_path,
        solved_model_path=solved_model_path,
        solved_reward=args.solved_reward,
        env_name=args.env,
        obs_space=obs_space,
        action_space=action_space,
        device=device,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=args.logdir,
        verbose=0,
        device=device,
        seed=args.seed,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        learning_rate=1e-4,
        ent_coef=0.05,
        n_epochs=4,
        batch_size=64,
        n_steps=512,
        policy_kwargs={"net_arch": [64, 64]},
        max_grad_norm=0.5,
    )

    try:
        model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
    finally:
        env.close()
        writer.close()

    model.save(args.save_path)
    stats = callback.final_stats()

    print("\n" + "=" * 70)
    print("Training Completed!")
    print("=" * 70)
    print(f"Final SB3 model saved to: {args.save_path}.zip")
    print(f"Total Episodes: {stats['episodes']}")
    print(f"Best Return: {stats['best_return']:.2f}")
    print(
        f"Average Return (last window): {stats['avg_recent_return']:.2f} "
        f"± {stats['std_recent_return']:.2f}"
    )
    if stats["solved"]:
        print(
            f"Solved! First solved episode: {stats['first_solved_episode']} "
            f"@ step {stats['first_solved_step']}"
        )
        print(f"Solved model saved to: {solved_model_path}.zip")
    else:
        print(f"Not solved yet (threshold {args.solved_reward:.1f})")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("TensorBoard Logging")
    print("=" * 70)
    print(f"Log directory: {logdir_abs}")
    print("\nTo view training plots, run:")
    print(f"  tensorboard --logdir {logdir_rel}")
    print("\nThen open your browser and navigate to:")
    print("  http://localhost:6006")
    print("\nOr use the full path:")
    print("  http://localhost:6006/#scalars")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
