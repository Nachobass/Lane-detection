# !pip install stable_baselines3
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rl_bridge.envs import make_env, UnityLaneEnv

HOST = "127.0.0.1"
# HOST = "localhost"
PORT = 5555
LOGDIR = "runs/ppo_lane"
TIMESTEPS = 50_000

os.makedirs(LOGDIR, exist_ok=True)

#env = make_env(HOST, PORT)
env = UnityLaneEnv(host=HOST, port=PORT) #make_env(HOST, PORT)
obs, _ = env.reset()
print("Obs shape:", obs.shape)

model = PPO(
    "MlpPolicy", env, verbose=1, tensorboard_log=LOGDIR,
    n_steps=1024, batch_size=256, gae_lambda=0.95, gamma=0.995, n_epochs=10,
    learning_rate=3e-4, clip_range=0.2,
)

rewards = []

def callback(_locals, _globals):
    return True

model.learn(total_timesteps=TIMESTEPS)

env.close()
print("Training finished.")

