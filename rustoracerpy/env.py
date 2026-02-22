from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from rustoracerpy.rustoracer import PySim


class RustoracerEnv(gym.Env):
    metadata: dict[str, list[str] | int] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        yaml: str,
        init_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
        max_steps: int = 10_000,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.render_mode: str | None = render_mode
        self._sim: PySim = PySim(yaml)
        self._pose: list[float] = list(init_pose)
        self._max_steps: int = max_steps
        self._steps: int = 0
        self.observation_space: spaces.Space = spaces.Box(
            0.0, self._sim.max_range, shape=(self._sim.n_beams,), dtype=np.float64
        )
        self.action_space: spaces.Space = spaces.Box(
            np.array([-0.4189, 0.0]), np.array([0.4189, 10.0]), dtype=np.float64
        )
        self.skeleton = self._sim.skeleton

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
        super().reset(seed=seed)
        if seed is not None:
            self._sim.seed(seed)
        self._steps = 0
        scan: NDArray[np.float64]
        state: NDArray[np.float64]
        scan, state, _ = self._sim.reset(self._pose)
        return scan, {"state": state}

    def step(
        self,
        action: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float, bool, bool, dict[str, NDArray[np.float64]]]:
        self._steps += 1
        scan: NDArray[np.float64]
        state: NDArray[np.float64]
        col: bool
        scan, state, col = self._sim.step(float(action[0]), float(action[1]))
        reward: float = -10.0 if col else float(action[1]) / 10
        return scan, reward, col, self._steps >= self._max_steps, {"state": state}

    def render(self) -> NDArray[np.uint8] | None:
        rgb: NDArray[np.uint8] = self._sim.render()
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

    def close(self) -> None:
        if hasattr(self, "_fig"):
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            del self._fig
