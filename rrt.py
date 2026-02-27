"""
Pure pursuit on skeleton spline + RRT local replan when obstacles block the path.
RRT only fires when forward LiDAR detects a close obstacle; otherwise pure pursuit
runs alone. A simple exponential filter on steering kills oscillation.
"""

import time
import numpy as np
from rustoracerpy import RustoracerEnv
import rerun as rr

WHEELBASE = 0.3302
STEER_FACTOR = 1.0 / 0.4189
LOOKAHEAD = 1.5
STEER_ALPHA = 0.35  # exponential smoothing (0=frozen, 1=instant)

SPEED_FAST = -0.6
SPEED_SLOW = -0.85
BRAKE_DIST = 2.0

RRT_TRIGGER = 1.5  # replan when forward clearance < this
RRT_ITERS = 200
RRT_STEP = 0.35
RRT_GOAL_BIAS = 0.30
CLEARANCE = 0.22  # car half-width + margin
EDGE_RES = 0.08

env = RustoracerEnv(yaml="maps/berlin.yaml", render_mode="human")
obs, info = env.reset()

N_BEAMS = env._sim.n_beams
FOV = env._sim.fov
waypoints = env.skeleton.reshape(-1, 2)
n_wps = len(waypoints)

beam_angles = np.linspace(-FOV / 2, FOV / 2, N_BEAMS)
forward_mask = np.abs(beam_angles) <= np.radians(60)

prev_steer = 0.0
rrt_path = None


def spline_goal(x, y):
    pos = np.array([x, y])
    ni = int(np.argmin(np.linalg.norm(waypoints - pos, axis=1)))
    for j in range(1, n_wps):
        idx = (ni + j) % n_wps
        if np.linalg.norm(waypoints[idx] - pos) >= LOOKAHEAD:
            return waypoints[idx]
    return waypoints[(ni + 1) % n_wps]


def pure_pursuit(x, y, theta, gx, gy):
    dx, dy = gx - x, gy - y
    local_y = -np.sin(theta) * dx + np.cos(theta) * dy
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return 0.0
    return float(
        np.clip(np.arctan(2.0 * local_y * WHEELBASE / L2) * STEER_FACTOR, -1, 1)
    )


def edge_free(a, b):
    d = np.linalg.norm(b - a)
    if d < 1e-9:
        return True
    pts = np.linspace(a, b, max(2, int(np.ceil(d / EDGE_RES))))
    return bool(np.all(env._sim.edt_at(pts.ravel()) >= CLEARANCE))


def rrt(start, goal):
    nodes = [start.copy()]
    parents = [-1]
    lo = np.minimum(start, goal) - 2.0
    hi = np.maximum(start, goal) + 2.0

    for _ in range(RRT_ITERS):
        sample = (
            goal if np.random.random() < RRT_GOAL_BIAS else np.random.uniform(lo, hi)
        )
        dists = np.linalg.norm(np.array(nodes) - sample, axis=1)
        ni = int(np.argmin(dists))
        diff = sample - nodes[ni]
        d = np.linalg.norm(diff)
        if d < 1e-9:
            continue
        new = nodes[ni] + diff / d * min(d, RRT_STEP)
        if not edge_free(nodes[ni], new):
            continue

        nodes.append(new)
        parents.append(ni)

        if np.linalg.norm(new - goal) < RRT_STEP:
            path, i = [], len(nodes) - 1
            while i >= 0:
                path.append(nodes[i])
                i = parents[i]
            path.reverse()
            return path
    return None


def pick_target(path, pos):
    for wp in path[1:]:
        if np.linalg.norm(wp - pos) >= RRT_STEP:
            return wp
    return path[-1]


def forward_clearance(scans):
    return float(scans[forward_mask].min())


def speed_for(clearance):
    if clearance <= 0.5:
        return SPEED_SLOW
    if clearance < BRAKE_DIST:
        return SPEED_SLOW + (clearance - 0.5) / (BRAKE_DIST - 0.5) * (
            SPEED_FAST - SPEED_SLOW
        )
    return SPEED_FAST


try:
    while True:
        t0 = time.perf_counter()
        x, y, theta, vel, *_ = info["state"][0]
        pos = np.array([x, y])
        scans = obs[0, :N_BEAMS]
        cl = forward_clearance(scans)
        goal = spline_goal(x, y)

        if cl < RRT_TRIGGER:
            result = rrt(pos, goal)
            if result and len(result) > 1:
                rrt_path = result
        else:
            rrt_path = None

        target = pick_target(rrt_path, pos) if rrt_path else goal
        raw_steer = pure_pursuit(x, y, theta, target[0], target[1])

        prev_steer = STEER_ALPHA * raw_steer + (1 - STEER_ALPHA) * prev_steer
        steer = float(np.clip(prev_steer, -1, 1))

        obs, _, term, trunc, info = env.step(np.array([[steer, speed_for(cl)]]))
        env.render()
        time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - t0)))

        if term[0] or trunc[0]:
            rrt_path, prev_steer = None, 0.0
            obs, info = env.reset()
except KeyboardInterrupt:
    pass
finally:
    env.close()
