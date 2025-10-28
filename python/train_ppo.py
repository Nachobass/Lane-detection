import argparse
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from rl_bridge.envs import make_env  # type: ignore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--frame_skip", type=int, default=1)
    p.add_argument("--logdir", default="runs/ppo_lane")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    def _env_fn():
        return make_env(args.host, args.port, frame_skip=args.frame_skip)

    env = DummyVecEnv([_env_fn])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=args.logdir,
        n_steps=1024,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.995,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
    )

    eval_env = _env_fn()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.logdir, "best"),
        log_path=args.logdir,
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=args.logdir, name_prefix="ppo_lane")

    model.learn(total_timesteps=args.timesteps, callback=[eval_callback, checkpoint_callback])

    model.save(os.path.join(args.logdir, "final_model"))

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()


