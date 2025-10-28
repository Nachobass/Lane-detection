import json
import socket
import time
from typing import Tuple, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class UnityLaneEnv(gym.Env):
    """Gymnasium environment wrapping the Unity RLBridgeServer over TCP.

    Observation: [s1, s2, s3, s4, s5, velocity] (float32)
    Action: [turn, throttle] in [-1, 1]
    Reward: delta in completion (0..1) per step, episode ends on crash or timeout.
    """

    metadata = {"render_modes": []}

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, frame_skip: int = 1, timeout_s: float = 5.0):
        super().__init__()
        self.host = host
        self.port = port
        self.frame_skip = frame_skip
        self.timeout_s = timeout_s

        self._sock: Optional[socket.socket] = None
        self._file_r = None
        self._file_w = None
        self._obs_size: Optional[int] = None

        # Spaces will be finalized after first reset when obs size is known
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    # --- Socket helpers ---
    def _connect(self):
        if self._sock is not None:
            return
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        self._file_r = self._sock.makefile(mode="r", encoding="utf-8", buffering=1, newline="\n")
        self._file_w = self._sock.makefile(mode="w", encoding="utf-8", buffering=1, newline="\n")

    def _send(self, obj: dict) -> dict:
        assert self._file_w is not None
        line = json.dumps(obj, separators=(",", ":"))
        self._file_w.write(line + "\n")
        self._file_w.flush()
        resp = self._file_r.readline()
        if not resp:
            raise RuntimeError("Unity bridge closed connection")
        try:
            data = json.loads(resp)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Bad JSON from Unity: {resp}") from e
        if "error" in data:
            raise RuntimeError(f"Unity error: {data['error']}")
        return data

    def _ensure_obs_space(self, obs: np.ndarray):
        if self._obs_size is None:
            self._obs_size = int(obs.shape[0])
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_size,), dtype=np.float32)

    # --- Gym API ---
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._connect()
        data = self._send({"cmd": "reset"})
        obs = np.array(data["obs"], dtype=np.float32)
        self._ensure_obs_space(obs)
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        data = None
        for _ in range(self.frame_skip):
            data = self._send({"cmd": "step", "action": action.tolist()})
        assert data is not None
        obs = np.array(data["obs"], dtype=np.float32)
        reward = float(data["reward"])
        terminated = bool(data["done"])  # crash or episode end
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            if self._file_w is not None:
                try:
                    self._send({"cmd": "close"})
                except Exception:
                    pass
        finally:
            try:
                self._file_w and self._file_w.close()
                self._file_r and self._file_r.close()
                self._sock and self._sock.close()
            finally:
                self._sock = None
                self._file_r = None
                self._file_w = None


def make_env(host: str = "127.0.0.1", port: int = 5555, frame_skip: int = 1):
    return UnityLaneEnv(host=host, port=port, frame_skip=frame_skip)




