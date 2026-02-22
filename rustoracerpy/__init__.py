from gymnasium.envs.registration import register
from rustoracerpy.env import RustoracerEnv

__all__ = ["RustoracerEnv"]

register(
    id="Rustoracer-v0",
    vector_entry_point="rustoracerpy.env:RustoracerEnv",
)
