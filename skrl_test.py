"""skrl PPO · RustoracerEnv (1024 native Rust agents)."""

import torch, torch.nn as nn
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
import rustoracerpy

env = wrap_env(
    rustoracerpy.RustoracerEnv("maps/berlin.yaml", num_envs=1024)
)  # 1024 agents in Rust
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Policy(GaussianMixin, Model):
    def __init__(self, o, a, d, **k):
        Model.__init__(self, o, a, d)
        GaussianMixin.__init__(self, min_log_std=-20, max_log_std=2)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role=""):
        return self.net(inputs["states"]), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    def __init__(self, o, a, d, **k):
        Model.__init__(self, o, a, d)
        DeterministicMixin.__init__(self)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

    def compute(self, inputs, role=""):
        return self.net(inputs["states"]), {}


RS = 32  # rollout steps
models = {
    "policy": Policy(env.observation_space, env.action_space, dev),
    "value": Value(env.observation_space, env.action_space, dev),
}
memory = RandomMemory(memory_size=RS, num_envs=env.num_envs, device=dev)
cfg = PPO_DEFAULT_CONFIG.copy()
cfg.update(
    {
        "rollouts": RS,
        "learning_epochs": 8,
        "mini_batches": 8,
        "discount_factor": 0.99,
        "lambda": 0.95,
        "learning_rate": 3e-4,
        "grad_norm_clip": 0.5,
    }
)
agent = PPO(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=dev,
)

SequentialTrainer(
    env=env, agents=agent, cfg={"timesteps": 1_000_000, "headless": True}
).train()
