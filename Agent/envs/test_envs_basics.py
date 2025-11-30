import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ConstantRewardEnv(gym.Env):
    def __init__(self):
        """Initialize ConstanRewardEnv"""
        super().__init__()
        self.observation_space = spaces.Discrete(1) # 0 constant observation
        self.action_space = spaces.Discrete(1)  # 1 action: 0 (reward = 1)

        self.done = False
        self.obs = 0

    def reset(self, seed=None, options=None):
        """Reset ConstanRewardEnv"""
        super().reset(seed=seed)
        self.done = False
        info = {}
        return self.obs, info

    def step(self, action):
        """Step ConstanRewardEnv"""
        assert self.action_space.contains(action), "Acción inválida"
        
        if self.done:
            raise RuntimeError("El episodio ya terminó. Llama a reset().")

        reward = 1

        self.done = True
        terminated = True
        truncated = False
        info = {}

        return self.obs, reward, terminated, truncated, info
    
    def render(self):
        """Render ConstanRewardEnv"""
        print("Entorno ConstanRewardEnv (observación siempre = 0)")


class RandomObsBinaryRewardEnv(gym.Env):
    def __init__(self):
        """Initialize RandomObsBinaryRewardEnv"""
        super().__init__()
        self.observation_space = spaces.Discrete(2) # -1 or 1
        self.action_space = spaces.Discrete(1)  # 1 action

        self.done = False
        self.obs = np.random.choice([-1, 1])

    def reset(self, seed=None, options=None):
        """Reset RandomObsBinaryRewardEnv"""
        super().reset(seed=seed)
        self.done = False
        self.obs = np.random.choice([-1, 1])
        info = {}
        return self.obs, info

    def step(self, action):
        """Step RandomObsBinaryRewardEnv"""
        assert self.action_space.contains(action), "Acción inválida"
        
        if self.done:
            raise RuntimeError("El episodio ya terminó. Llama a reset().")

        reward = self.obs 

        self.done = True
        terminated = True
        truncated = False
        info = {}

        return self.obs, reward, terminated, truncated, info
    
    def render(self):
        """Render RandomObsBinaryRewardEnv"""
        print(f"Observación: {self.obs}, Recompensa: {self.obs}")


class TwoStepDelayedRewardEnv(gym.Env):
    def __init__(self):
        """Initialize TwoStepDelayedRewardEnv"""
        super().__init__()
        self.observation_space = spaces.Discrete(2) # 0 in the first step, 1 in the second step
        self.action_space = spaces.Discrete(1)  # 1 action

        self.done = False
        self.obs = 0
        self.step_count = 0

    def reset(self, seed=None, options=None):
        """Reset TwoStepDelayedRewardEnv"""
        super().reset(seed=seed)
        self.done = False
        self.step_count = 0
        info = {}
        return self.obs, info

    def step(self, action):
        """Step TwoStepDelayedRewardEnv"""
        assert self.action_space.contains(action), "Acción inválida"

        if self.done:
            raise RuntimeError("El episodio ya terminó. Llama a reset().")

        if self.step_count == 0:
            self.obs = 0
            reward = 0
            terminated = False
        elif self.step_count == 1:
            self.obs = 1
            reward = 1
            terminated = True
        else:
            raise RuntimeError("El episodio ya terminó. Llama a reset().")

        truncated = False
        self.done = terminated
        info = {}
        
        self.step_count += 1

        return self.obs, reward, terminated, truncated, info
    
    def render(self):
        """Render TwoStepDelayedRewardEnv"""
        print(f"Step: {self.step_count}, Obs: {self.obs}, Done: {self.done}")