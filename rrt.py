import time
import threading

import numpy as np
import rerun as rr

from rustoracerpy import RustoracerEnv

WHEELBASE = 0.3302
STEER_FACTOR = 1.0 / 0.4189
LOOKAHEAD = 3
STEER_ALPHA = 0.35

SPEED_FAST = -0.6
SPEED_SLOW = -0.85
BRAKE_DIST = 2.0

RRT_ITERS = 200
RRT_STEP = 0.35
RRT_GOAL_BIAS = 0.30
CLEARANCE = 0.22
EDGE_RES = 0.08

OBS_RADIUS = 0.35

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

obstacles: list[np.ndarray] = []
place_flag = threading.Event()
clear_flag = threading.Event()


def _input_listener():
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().lower() == "c":
            clear_flag.set()
        else:
            place_flag.set()


threading.Thread(target=_input_listener, daemon=True).start()
print("── Press ENTER to place an obstacle at the lookahead point ──")
print("── Type 'c' + ENTER to clear all obstacles                 ──")


def _log_obstacles():
    """Send all obstacle circles to Rerun."""
    if not obstacles:
        rr.log("world/obstacles", rr.Clear(recursive=False))
        return
    centres_world = np.array(obstacles)  # (N, 2)
    centres_px = env._sim.world_to_pixels(
        centres_world.ravel().astype(np.float64)
    ).reshape(-1, 2)
    rr.log(
        "world/obstacles",
        rr.Points2D(
            centres_px,
            radii=[10.0] * len(centres_px),
            colors=[[255, 50, 50, 200]] * len(centres_px),
        ),
    )


def _hits_obstacle(pts: np.ndarray) -> bool:
    for o in obstacles:
        if np.any(np.linalg.norm(pts - o, axis=1) < OBS_RADIUS + CLEARANCE):
            return True
    return False


def _edt_clear(pts):
    flat = np.ascontiguousarray(pts.ravel(), dtype=np.float64)
    if not bool(np.all(np.asarray(env._sim.edt_at(flat)) >= CLEARANCE)):
        return False
    return not _hits_obstacle(pts)


def edge_free(a, b):
    d = np.linalg.norm(b - a)
    if d < 1e-9:
        return _edt_clear(a.reshape(1, 2))
    n = max(2, int(np.ceil(d / EDGE_RES)))
    pts = np.column_stack([np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)])
    return _edt_clear(pts)


def spline_goal(x, y):
    pos = np.array([x, y])
    nearest = int(np.argmin(np.linalg.norm(waypoints - pos, axis=1)))
    for j in range(1, n_wps):
        idx = (nearest + j) % n_wps
        if np.linalg.norm(waypoints[idx] - pos) >= LOOKAHEAD and _edt_clear(
            np.array([waypoints[idx]])
        ):
            return waypoints[idx]
    return waypoints[(nearest + 1) % n_wps]


def pure_pursuit(x, y, theta, gx, gy):
    dx, dy = gx - x, gy - y
    local_y = -np.sin(theta) * dx + np.cos(theta) * dy
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return 0.0
    curvature = np.arctan(2.0 * local_y * WHEELBASE / L2) * STEER_FACTOR
    return float(np.clip(curvature, -1, 1))


def _to_px(pt):
    return env._sim.world_to_pixels(np.array(pt, dtype=np.float64)).reshape(-1, 2)


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

        if not _edt_clear(new.reshape(1, 2)) or not edge_free(nodes[ni], new):
            continue

        nodes.append(new)
        parents.append(ni)

        if np.linalg.norm(new - goal) < RRT_STEP:
            path, i = [], len(nodes) - 1
            while i >= 0:
                path.append(nodes[i])
                i = parents[i]
            path.reverse()

            edges = [
                np.vstack([_to_px(nodes[parents[j]]), _to_px(nodes[j])])
                for j in range(1, len(nodes))
            ]
            if edges:
                rr.log("world/RRT_graph", rr.LineStrips2D(edges))

            path_px = [_to_px(p) for p in path]
            if len(path_px) > 1:
                segments = [
                    np.vstack([path_px[k], path_px[k + 1]])
                    for k in range(len(path_px) - 1)
                ]
                rr.log(
                    "world/RRT_path",
                    rr.LineStrips2D(segments, colors=[[0, 120, 255, 255]]),
                )

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
        t = (clearance - 0.5) / (BRAKE_DIST - 0.5)
        return SPEED_SLOW + t * (SPEED_FAST - SPEED_SLOW)
    return SPEED_FAST


try:
    rng = np.random.default_rng()
    while True:
        t0 = time.perf_counter()
        x, y, theta, vel, *_ = info["state"][0]
        pos = np.array([x, y])
        scans = obs[0, :N_BEAMS]
        cl = forward_clearance(scans)
        goal = spline_goal(x, y)

        if place_flag.is_set():
            place_flag.clear()
            obstacles.append(goal.copy() + rng.random(size=(2)))
            print(
                f"  ✦ Obstacle placed at ({goal[0]:.2f}, {goal[1]:.2f})  "
                f"[total: {len(obstacles)}]"
            )

        if clear_flag.is_set():
            clear_flag.clear()
            obstacles.clear()
            rr.log("world/obstacles", rr.Clear(recursive=False))
            print("  ✦ All obstacles cleared")

        _log_obstacles()

        goal_px = _to_px(goal)
        rr.log(
            "world/lookahead",
            rr.Points2D(goal_px, radii=[5.0], colors=[[255, 200, 0, 160]]),
        )

        result = rrt(pos, goal)
        if result and len(result) > 1:
            rrt_path = result

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
