import numpy as np
import rustoracerpy
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecMonitor, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
from rustoracerpy.env import RustoracerEnv


class SB3VecAdapter(VecEnv):
    """Adapt a gymnasium VectorEnv to SB3's VecEnv interface."""

    def __init__(self, venv: RustoracerEnv):
        self.venv = venv
        super().__init__(
            num_envs=venv.num_envs,
            observation_space=venv.single_observation_space,
            action_space=venv.single_action_space,
        )
        self._actions: np.ndarray | None = None

    def reset(self) -> np.ndarray:
        obs, _info = self.venv.reset()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        obs, rewards, terminated, truncated, infos = self.venv.step(self._actions)
        dones = terminated | truncated
        info_list = [{} for _ in range(self.num_envs)]
        return obs, rewards, dones, info_list

    def close(self) -> None:
        self.venv.close()

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.venv, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.venv, attr_name, value)

    def render(self, mode="rgb_array"):
        return self.venv.render()

    def seed(self, seed=None):
        if seed is not None:
            self.venv._sim.seed(seed)


NUM_ENVS = 16

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1e10,
    "env_name": "Rustoracer-v0",
}
run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
)

env = SB3VecAdapter(
    RustoracerEnv(yaml="maps/berlin.yaml", num_envs=NUM_ENVS, render_mode="rgb_array")
)
env = VecMonitor(env)
env = VecVideoRecorder(
    env,
    f"videos/{run.id}",
    record_video_trigger=lambda x: x % 10_000 == 0,
    video_length=1_000,
)
model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)
run.finish()
