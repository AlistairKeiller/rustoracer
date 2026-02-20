import gymnasium as gym
import numpy as np
import rustoracerpy

env = gym.make("Rustoracer-v0", yaml="maps/berlin.yaml", render_mode="human")
obs, info = env.reset()

# Get skeleton waypoints and sort them into a path via nearest-neighbor greedy ordering
raw_pts = env.unwrapped.skeleton  # (N, 2) unordered centerline points

# Greedy nearest-neighbor ordering starting from the closest point to the car
pose = info["pose"]
dists = np.linalg.norm(raw_pts - np.array([pose[0], pose[1]]), axis=1)
order = [int(np.argmin(dists))]
remaining = set(range(len(raw_pts)))
remaining.discard(order[0])
while remaining:
    last = raw_pts[order[-1]]
    rem_idx = np.array(list(remaining))
    d = np.linalg.norm(raw_pts[rem_idx] - last, axis=1)
    nearest = rem_idx[np.argmin(d)]
    order.append(int(nearest))
    remaining.discard(nearest)
waypoints = raw_pts[order]  # ordered (M, 2)

# Pure pursuit parameters
LOOKAHEAD = 1.5  # metres
WHEELBASE = 0.15875 + 0.17145  # front + rear axle lengths from car.rs
SPEED = 5.0  # constant forward speed

for _ in range(10_000):
    x, y, theta = info["pose"]
    pos = np.array([x, y])

    # Find closest waypoint index
    dists = np.linalg.norm(waypoints - pos, axis=1)
    nearest_idx = int(np.argmin(dists))

    # Walk forward along the path to find the lookahead point
    goal = waypoints[nearest_idx]
    for j in range(1, len(waypoints)):
        idx = (nearest_idx + j) % len(waypoints)
        if np.linalg.norm(waypoints[idx] - pos) >= LOOKAHEAD:
            goal = waypoints[idx]
            break

    # Transform goal to vehicle frame
    dx = goal[0] - x
    dy = goal[1] - y
    local_x = np.cos(theta) * dx + np.sin(theta) * dy
    local_y = -np.sin(theta) * dx + np.cos(theta) * dy

    # Pure pursuit steering law
    L = np.hypot(local_x, local_y)
    if L < 1e-6:
        steer = 0.0
    else:
        curvature = 2.0 * local_y / (L * L)
        steer = float(np.clip(np.arctan(curvature * WHEELBASE), -0.4189, 0.4189))

    obs, reward, terminated, truncated, info = env.step(np.array([steer, SPEED]))
    env.render()

    if terminated or truncated:
        obs, info = env.reset()

env.close()
