from gymnasium.envs.registration import register

register(
    id="Rustoracer-v0",
    entry_point="rustoracerpy.env:RustoracerEnv",
)
