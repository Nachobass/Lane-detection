# import argparse
# import os
# import time
# from typing import Dict, Any

# import optuna
# from optuna.pruners import MedianPruner
# from optuna.samplers import TPESampler

# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.callbacks import EvalCallback

# from rl_bridge.envs import make_env  # type: ignore


# def suggest_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
#     params: Dict[str, Any] = {
#         "n_steps": trial.suggest_categorical("n_steps", [256, 512, 1024, 2048]),
#         "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
#         "gamma": trial.suggest_float("gamma", 0.95, 0.9999, log=True),
#         "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.98),
#         "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True),
#         "clip_range": trial.suggest_float("clip_range", 0.1, 0.35),
#         "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.02, log=False),
#         "vf_coef": trial.suggest_float("vf_coef", 0.3, 1.0),
#         "n_epochs": trial.suggest_categorical("n_epochs", [5, 10, 15]),
#         "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
#     }
#     # Keep batch_size <= n_steps as recommended by SB3
#     if params["batch_size"] > params["n_steps"]:
#         params["batch_size"] = params["n_steps"]
#     return params


# def make_vec(host: str, port: int, frame_skip: int):
#     def _fn():
#         # give Unity a moment on first connect per process
#         env_fn = make_env(host, port, frame_skip=frame_skip)
#         return env_fn()
#     return DummyVecEnv([_fn])


# def objective(trial: optuna.Trial, args) -> float:
#     # Unique log dir per trial for TensorBoard and checkpoints
#     trial_logdir = os.path.join(args.logdir, f"trial_{trial.number}")
#     os.makedirs(trial_logdir, exist_ok=True)

#     # Create train env
#     env = make_vec(args.host, args.port, args.frame_skip)

#     # Model with suggested hyperparams
#     params = suggest_ppo_params(trial)
#     model = PPO(
#         "MlpPolicy",
#         env,
#         verbose=0,
#         tensorboard_log=trial_logdir,
#         **params,
#     )

#     # Eval env and callback (deterministic evaluation)
#     eval_env = make_vec(args.host, args.port, args.frame_skip)
#     eval_callback = EvalCallback(
#         eval_env,
#         best_model_save_path=trial_logdir,
#         log_path=trial_logdir,
#         eval_freq=max(10_000, args.timesteps // 10),
#         n_eval_episodes=5,
#         deterministic=True,
#         render=False,
#     )

#     # Small delay to reduce reset race on cold Unity
#     time.sleep(0.5)

#     try:
#         model.learn(total_timesteps=args.timesteps, callback=eval_callback, progress_bar=False)
#         # Load best mean reward recorded by EvalCallback
#         # EvalCallback writes evaluations.npz with mean rewards; we can use its attribute
#         mean_reward = float(eval_callback.best_mean_reward)
#         # Optuna tries to maximize objective
#         return mean_reward
#     finally:
#         try:
#             env.close()
#         except Exception:
#             pass
#         try:
#             eval_env.close()
#         except Exception:
#             pass


# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--host", default="127.0.0.1")
#     p.add_argument("--port", type=int, default=5555)
#     p.add_argument("--frame_skip", type=int, default=1)
#     p.add_argument("--logdir", default="runs/ppo_lane/tuning")
#     p.add_argument("--timesteps", type=int, default=100_000, help="Timesteps per trial")
#     p.add_argument("--trials", type=int, default=20)
#     p.add_argument("--seed", type=int, default=0)
#     p.add_argument("--study_name", default="ppo_lane_opt")
#     p.add_argument("--storage", default=None, help="Optuna storage URL (e.g., sqlite:///ppo_opt.db)")
#     return p.parse_args()


# def main():
#     args = parse_args()
#     os.makedirs(args.logdir, exist_ok=True)

#     sampler = TPESampler(seed=args.seed)
#     pruner = MedianPruner(n_startup_trials=min(5, args.trials // 4))

#     study = optuna.create_study(
#         study_name=args.study_name,
#         direction="maximize",
#         sampler=sampler,
#         pruner=pruner,
#         storage=args.storage,
#         load_if_exists=bool(args.storage),
#     )

#     study.optimize(lambda t: objective(t, args), n_trials=args.trials, gc_after_trial=True)

#     print("Best trial:", study.best_trial.number)
#     print("  Value (mean reward):", study.best_value)
#     print("  Params:")
#     for k, v in study.best_trial.params.items():
#         print(f"    {k}: {v}")


# if __name__ == "__main__":
#     main()


