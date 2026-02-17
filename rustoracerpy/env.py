from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rustoracerpy.rustoracer import PySim


class RustoracerEnv(gym.Env):
    def __init__(self, yaml: str, init_pose=(0.0, 0.0, 0.0), max_steps=10_000):
        super().__init__()
        self._sim = PySim(yaml)
        self._pose = list(init_pose)
        self._max_steps = max_steps
        self._steps = 0

        self.observation_space = spaces.Box(
            0.0, self._sim.max_range, shape=(self._sim.n_beams,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            np.array([-0.4189, 0.0]), np.array([0.4189, 7.0]), dtype=np.float64
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        scan, pose, _ = self._sim.reset(self._pose)
        return scan, {"pose": pose}

    def step(self, action):
        self._steps += 1
        scan, pose, col = self._sim.step(float(action[0]), float(action[1]))
        reward = -100.0 if col else float(action[1])
        return scan, reward, col, self._steps >= self._max_steps, {"pose": pose}

    def close(self):
        pass
