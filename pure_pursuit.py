import gymnasium as gym
import numpy as np
import rustoracerpy

LOOKAHEAD = 1.5
WHEELBASE = 0.15875 + 0.17145
SPEED = 5.0

env = gym.make("Rustoracer-v0", yaml="maps/berlin.yaml", render_mode="human")
obs, info = env.reset()
waypoints = env.unwrapped.skeleton(info["pose"])

while True:
    x, y, theta = info["pose"]
    pos = np.array([x, y])

    # Find lookahead point
    nearest = int(np.argmin(np.linalg.norm(waypoints - pos, axis=1)))
    goal = waypoints[nearest]
    for j in range(1, len(waypoints)):
        idx = (nearest + j) % len(waypoints)
        if np.linalg.norm(waypoints[idx] - pos) >= LOOKAHEAD:
            goal = waypoints[idx]
            break

    # Pure pursuit steering
    dx, dy = goal[0] - x, goal[1] - y
    local_y = -np.sin(theta) * dx + np.cos(theta) * dy
    L2 = dx * dx + dy * dy
    steer = (
        float(np.clip(np.arctan(2.0 * local_y * WHEELBASE / L2), -0.4189, 0.4189))
        if L2 > 1e-12
        else 0.0
    )

    obs, reward, terminated, truncated, info = env.step(np.array([steer, SPEED]))
    env.render()
    if terminated or truncated:
        obs, info = env.reset()

env.close()
