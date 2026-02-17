from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rustoracerpy.rustoracer import PySim


class RustoracerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self, yaml: str, init_pose=(0.0, 0.0, 0.0), max_steps=10_000, render_mode=None
    ):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._sim = PySim(yaml)
        self._pose = list(init_pose)
        self._max_steps = max_steps
        self._steps = 0

        self.observation_space = spaces.Box(
            0.0, self._sim.max_range, shape=(self._sim.n_beams,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            np.array([-0.4189, 0.0]), np.array([0.4189, 10.0]), dtype=np.float64
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        scan, pose, _ = self._sim.reset(self._pose)
        return scan, {"pose": pose}

    def step(self, action):
        self._steps += 1
        scan, pose, col = self._sim.step(float(action[0]), float(action[1]))
        reward = -10.0 if col else float(action[1]) / 10
        return scan, reward, col, self._steps >= self._max_steps, {"pose": pose}

    def render(self):
        rgb = self._sim.render()
        if self.render_mode == "human":
            import matplotlib.pyplot as plt

            if not hasattr(self, "_fig"):
                plt.ion()
                self._fig, self._ax = plt.subplots()
                self._img = self._ax.imshow(rgb)
            else:
                self._img.set_data(rgb)
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
            return None
        return rgb

    def close(self):
        if hasattr(self, "_fig"):
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            del self._fig
