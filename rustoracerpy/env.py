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
        "render_fps": 30,
    }

    def __init__(
        self,
        yaml: str,
        num_envs: int = 1,
        max_steps: int = 10_000,
        render_mode: str | None = None,
    ) -> None:
        self._sim: PySim = PySim(yaml, num_envs)
        self._max_steps: int = max_steps
        self._steps: NDArray[np.int32] = np.zeros(num_envs, dtype=np.int32)
        self._prev_progress: NDArray[np.float64] = np.zeros(num_envs, dtype=np.float64)

        single_obs_space = spaces.Box(
            0.0, self._sim.max_range, shape=(self._sim.n_beams,), dtype=np.float64
        )
        single_act_space = spaces.Box(
            np.array([-0.4189, 0.0]), np.array([0.4189, 20.0]), dtype=np.float64
        )

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
        self._steps[:] = 0
        self._prev_progress[:] = 0.0
        scans, states, _, progress = self._sim.reset()
        return scans.reshape(self.num_envs, -1), {
            "state": states.reshape(self.num_envs, -1),
            "progress": progress,
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
        self._steps += 1
        scans, states, cols, progress = self._sim.step(actions.ravel())
        obs = scans.reshape(self.num_envs, -1)

        dp = progress - self._prev_progress
        rewards = np.where(cols, -100.0, dp * 100.0 + actions[:, 1] * 0.001 - 0.001)
        self._prev_progress = progress.copy()

        terminated = cols
        truncated = self._steps >= self._max_steps
        dones = terminated | truncated

        for i in range(self.num_envs):
            if dones[i]:
                self._sim.reset_single(i)
                self._steps[i] = 0
                self._prev_progress[i] = 0.0
                new_scans, new_states, _, new_progress = self._sim.observe()
                new_scans = new_scans.reshape(self.num_envs, -1)
                new_states = new_states.reshape(self.num_envs, -1)
                obs[i] = new_scans[i]
                states = states.reshape(self.num_envs, -1)
                states[i] = new_states[i]
                progress[i] = new_progress[i]

        return (
            obs,
            rewards,
            terminated,
            truncated,
            {
                "state": states.reshape(self.num_envs, -1),
                "progress": progress,
            },
        )

    def render(self) -> NDArray[np.uint8] | None:
        if self.render_mode == "rgb_array":
            return self._sim.render()
        return None

    def close(self) -> None:
        pass
