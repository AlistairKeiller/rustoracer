from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from gymnasium.vector.utils import batch_space
from rustoracerpy.rustoracer import PySim


class RustoracerEnv(gym.vector.VectorEnv):
    metadata: dict[str, list[str] | int] = {
        "render_modes": ["rgb_array"],
        "render_fps": 60,
    }

    def __init__(
        self,
        yaml: str,
        num_envs: int = 1,
        max_steps: int = 10_000,
        render_mode: str | None = None,
    ) -> None:
        self._sim: PySim = PySim(yaml, num_envs, max_steps)

        single_obs_space = spaces.Box(
            np.array([0.0] * self._sim.n_beams + [-5.0, -0.5]),
            np.array([self._sim.max_range] * self._sim.n_beams + [20.0, 0.5]),
            dtype=np.float64,
        )
        single_act_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float64)

        self.num_envs = num_envs
        self.render_mode = render_mode
        self.single_observation_space = single_obs_space
        self.single_action_space = single_act_space
        self.observation_space = batch_space(single_obs_space, num_envs)
        self.action_space = batch_space(single_act_space, num_envs)

        self.skeleton: NDArray[np.float64] = self._sim.skeleton

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
        if seed is not None:
            self._sim.seed(seed)
        scans, _rewards, _terminated, _truncated, states = self._sim.reset()
        return scans.reshape(self.num_envs, -1), {
            "state": states.reshape(self.num_envs, -1),
        }

    def step(
        self,
        actions: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
        NDArray[np.bool_],
        dict[str, NDArray],
    ]:
        scans, rewards, terminated, truncated, states = self._sim.step(actions.ravel())
        return (
            scans.reshape(self.num_envs, -1),
            rewards,
            terminated,
            truncated,
            {
                "state": states.reshape(self.num_envs, -1),
            },
        )

    def render(self) -> NDArray[np.uint8] | None:
        if self.render_mode == "rgb_array":
            return self._sim.render()
        return None

    def close(self) -> None:
        pass
