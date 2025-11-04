## PPO training with Unity lane simulator

### 1) In Unity
- Open the project at `UnityProject`.
- Add the `RLBridgeServer` MonoBehaviour to an active GameObject in the training scene (e.g., `Main.unity`).
- Ensure `TrackManager` and a `CarController` prototype are present.
- Press Play; the Console should show `RLBridgeServer listening on 127.0.0.1:5555`.

### 2) Python setup
```bash
cd python
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Quick check
```python
from rl_bridge.envs.unity_lane_env import make_env
env = make_env()
obs, _ = env.reset()
print(obs.shape)
obs, r, term, trunc, info = env.step([0.0, 0.2])
print(r, term)
env.close()
```

### 4) Train PPO
```bash
python train_ppo.py --timesteps 200000 --logdir runs/ppo_lane
```

### 5) Notebook
Open `notebooks/ppo_experiments.ipynb` for an end-to-end walkthrough and analysis.




### 6) Test with CartPole
First, train the model:
```bash
python PPO/train_cartpole_ppo.py --timesteps 100000 --logdir runs/cartpole_ppo_clip
```

Then, evaluate the model:
```bash
python PPO/eval_cartpole_ppo.py --weights runs/cartpole_ppo_clip/final.pt --episodes 10
```

To evaluate the model with rendering, add the `--render` flag:
```bash
python PPO/eval_cartpole_ppo.py --weights runs/cartpole_ppo_clip/final.pt --episodes 10 --render
```

### 7) Test with BipedalWalker
First, train the model:
```bash
python PPO/train_bipedalwalker_ppo.py --timesteps 2000000 --logdir runs/bipedalwalker_ppo_clip
```

Then, evaluate the model:
```bash
python PPO/eval_bipedalwalker_ppo.py --weights runs/bipedalwalker_ppo_clip/final.pt --episodes 10
```

To evaluate the model with rendering, add the `--render` flag:
```bash
python PPO/eval_bipedalwalker_ppo.py --weights runs/bipedalwalker_ppo_clip/final.pt --episodes 10 --render
```