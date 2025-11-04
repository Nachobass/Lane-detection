import argparse
import torch
import numpy as np

from rl_bridge.envs.unity_lane_env import UnityLaneEnv
from PPO.ppo_clip import PPOClip

HOST = "127.0.0.1"
PORT = 5555


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=HOST)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--frame_skip", type=int, default=1)      # frame_skip es cuantas veces se ejecuta el step del env
    p.add_argument("--weights", required=True, help="Path to saved .pt weights")
    p.add_argument("--episodes", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = UnityLaneEnv(host=args.host, port=args.port, frame_skip=args.frame_skip)
    obs, _ = env.reset()
    agent = PPOClip.load(args.weights, obs_dim=int(obs.shape[0]), act_dim=2, device=device)

    for ep in range(args.episodes):
        done = False
        ret = 0.0
        steps = 0
        obs, _ = env.reset()
        while not done and steps < 5000:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, _, _ = agent.select_action(obs_t)
            obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            ret += float(reward)
            done = bool(terminated) or bool(truncated)
            steps += 1
        print(f"Episode {ep+1}: return={ret:.2f}, steps={steps}, done={done}")

    env.close()


if __name__ == "__main__":
    main()


