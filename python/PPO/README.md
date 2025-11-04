PPO options
===========

Prereqs
-------
- Unity app running with `RLBridgeServer` on 127.0.0.1:5555 (press Play in `Main.unity`).
- Install deps:

```bash
pip install -r ../requirements.txt
```

Stable-Baselines3 tuner
-----------------------

```bash
python PPO/tune_ppo.py \
  --host 127.0.0.1 --port 5555 \
  --timesteps 100000 \
  --trials 20 \
  --logdir runs/ppo_lane/tuning
```

Optional persistent storage (resume/inspect best):

```bash
python PPO/tune_ppo.py --storage sqlite:///ppo_opt.db --trials 40
```

TensorBoard
-----------

Each trial writes logs to a subfolder under `runs/ppo_lane/tuning/trial_<N>`. Launch TensorBoard from the project root:

```bash
tensorboard --logdir runs/ppo_lane/tuning
```

Notes
-----
- The objective is the best mean reward from `EvalCallback` over 5 episodes.
- You can adjust the search space in `suggest_ppo_params`.

Custom PPO-Clip (from scratch)
------------------------------

Train:

```bash
python PPO/train_custom_ppo.py --host 127.0.0.1 --port 5555 \
  --timesteps 200000 --logdir runs/custom_ppo_clip
```

Saving and loading
------------------

- The trainer writes periodic checkpoints to `runs/custom_ppo_clip/ckpt_*.pt`, a best model to `runs/custom_ppo_clip/best.pt`, and a final model to `runs/custom_ppo_clip/final.pt` (path configurable via `--save_path`).

- Evaluate a saved model (no training):

```bash
python PPO/eval_custom_ppo.py --weights runs/custom_ppo_clip/final.pt \
  --host 127.0.0.1 --port 5555 --episodes 5
```

Tune with Optuna:

```bash
python PPO/tune_custom_ppo.py --host 127.0.0.1 --port 5555 \
  --timesteps 100000 --trials 20 --logdir runs/custom_ppo_clip/tuning
```

TensorBoard:

```bash
tensorboard --logdir runs/custom_ppo_clip
```

