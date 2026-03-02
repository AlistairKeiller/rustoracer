import time
import numpy as np
from rustoracerpy import RustoracerEnv

for n_envs in [1, 16, 64, 256, 1024]:
    env = RustoracerEnv("maps/berlin.yaml", num_envs=n_envs, max_steps=10_000)
    env.reset(seed=42)
    actions = np.random.uniform(-1, 1, (n_envs, 2))

    # Warmup
    for _ in range(10):
        env.step(actions)

    t0 = time.perf_counter()
    N = 1000
    for _ in range(N):
        env.step(actions)
    dt = time.perf_counter() - t0

    sps = n_envs * N / dt
    print(f"n_envs={n_envs:>5}  {sps:>12,.0f} steps/s  {dt / N * 1000:.2f} ms/step")
