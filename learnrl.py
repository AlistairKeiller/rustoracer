from __future__ import annotations

import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import math
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import tensordict
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import tyro
import wandb
from tensordict import from_module
from tensordict.nn import CudaGraphModule
from torch.distributions.normal import Normal

from rustoracerpy import RustoracerEnv


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    yaml: str = "maps/berlin.yaml"
    """path to the RustoracerEnv YAML map file"""
    total_timesteps: int = 50_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1024
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    measure_burnin: int = 1
    """Number of burn-in iterations for speed measure."""

    compile: bool = False
    """whether to use torch.compile."""
    cudagraphs: bool = False
    """whether to use cudagraphs on top of compile."""

    video_interval: int = 5
    """record an evaluation video every N iterations (0 to disable)"""
    video_max_steps: int = 1000
    """max steps per evaluation video episode"""


class RunningMeanStd:
    """Welford online estimator (same algorithm as gym.wrappers.NormalizeObservation)."""

    def __init__(self, shape=(), eps=1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / tot) / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, n_obs, n_act, device=None):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, 64, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1, device=device), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_obs, 64, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_act, device=device), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean + action_std * torch.randn_like(action_mean)
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(obs),
        )


def gae(next_obs, next_done, container):
    next_value = get_value(next_obs).reshape(-1)
    lastgaelam = 0
    nextnonterminals = (~container["dones"]).float().unbind(0)
    vals = container["vals"]
    vals_unbind = vals.unbind(0)
    rewards = container["rewards"].unbind(0)

    advantages = []
    nextnonterminal = (~next_done).float()
    nextvalues = next_value
    for t in range(args.num_steps - 1, -1, -1):
        cur_val = vals_unbind[t]
        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - cur_val
        advantages.append(
            delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        )
        lastgaelam = advantages[-1]

        nextnonterminal = nextnonterminals[t]
        nextvalues = cur_val

    advantages = container["advantages"] = torch.stack(list(reversed(advantages)))
    container["returns"] = advantages + vals
    return container


def rollout(obs, done, avg_returns=[]):
    ts = []
    for step in range(args.num_steps):
        action, logprob, _, value = policy(obs=obs)

        next_obs, reward, next_done, infos = step_func(action)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    r = float(info["episode"]["r"])
                    avg_returns.append(r)

        ts.append(
            tensordict.TensorDict._new_unsafe(
                obs=obs,
                dones=done,
                vals=value.flatten(),
                actions=action,
                logprobs=logprob,
                rewards=reward,
                batch_size=(args.num_envs,),
            )
        )

        obs = next_obs = next_obs.to(device, non_blocking=True)
        done = next_done.to(device, non_blocking=True)

    container = torch.stack(ts, 0).to(device)
    return next_obs, done, container


def update(obs, actions, logprobs, advantages, returns, vals):
    optimizer.zero_grad()
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs, actions)
    logratio = newlogprob - logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = vals + torch.clamp(
            newvalue - vals,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    loss.backward()
    gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()

    return (
        approx_kl,
        v_loss.detach(),
        pg_loss.detach(),
        entropy_loss.detach(),
        old_approx_kl,
        clipfrac,
        gn,
    )


update = tensordict.nn.TensorDictModule(
    update,
    in_keys=["obs", "actions", "logprobs", "advantages", "returns", "vals"],
    out_keys=[
        "approx_kl",
        "v_loss",
        "pg_loss",
        "entropy_loss",
        "old_approx_kl",
        "clipfrac",
        "gn",
    ],
)


# ─────────────────────────────────────────────────────────────
# Video recording helper  (FIX: render_mode="rgb_array")
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def record_eval_video(
    yaml: str,
    agent_inference: Agent,
    obs_rms: RunningMeanStd,
    device: torch.device,
    max_steps: int = 1000,
) -> Tuple[wandb.Video | None, float]:
    """Run one deterministic episode in a single-env copy, return (wandb.Video, total_reward)."""
    eval_env = RustoracerEnv(
        yaml=yaml,
        num_envs=1,
        max_steps=max_steps,
        render_mode="rgb_array",
    )
    raw_obs, _ = eval_env.reset(seed=42)

    frames: list[np.ndarray] = []
    total_reward = 0.0

    for _ in range(max_steps):
        # render BEFORE taking the step so frame 0 is the initial state
        frame = eval_env.render()  # (H, W, 3) uint8
        if frame is not None:
            frames.append(frame)

        obs_norm = np.clip(obs_rms.normalize(raw_obs), -10, 10)
        obs_t = torch.tensor(obs_norm, device=device, dtype=torch.float)

        # deterministic: use mean action
        action_mean = agent_inference.actor_mean(obs_t)
        action_np = action_mean.cpu().numpy().astype(np.float64).clip(-1.0, 1.0)

        raw_obs, reward, terminated, truncated, _ = eval_env.step(action_np)
        total_reward += float(reward[0])

        if terminated[0] or truncated[0]:
            frame = eval_env.render()
            if frame is not None:
                frames.append(frame)
            break

    eval_env.close()

    if len(frames) == 0:
        return None, total_reward

    # wandb.Video expects (T, C, H, W) uint8
    video_np = np.stack(frames, axis=0)  # (T, H, W, 3)
    video_np = video_np.transpose(0, 3, 1, 2)  # (T, C, H, W)
    video = wandb.Video(video_np, fps=30, format="mp4")
    return video, total_reward


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = tyro.cli(Args)

    batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = batch_size // args.num_minibatches
    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = (
        f"Rustoracer__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}"
    )

    print(f"batch_size        = {args.batch_size:,}")
    print(f"num_iterations    = {args.num_iterations}")
    print(f"video_interval    = {args.video_interval}")

    wandb.init(
        project="ppo_continuous_action",
        name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
        config=vars(args),
        save_code=True,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ####### Environment setup #######
    envs = RustoracerEnv(
        yaml=args.yaml,
        num_envs=args.num_envs,
    )
    n_act = math.prod(envs.single_action_space.shape)
    n_obs = math.prod(envs.single_observation_space.shape)
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    obs_rms = RunningMeanStd(shape=(n_obs,))
    ret_rms = RunningMeanStd(shape=())
    disc_returns = np.zeros(args.num_envs, dtype=np.float64)
    ep_returns = np.zeros(args.num_envs, dtype=np.float64)

    def step_func(
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        act_np = action.cpu().numpy().astype(np.float64).clip(-1.0, 1.0)

        next_obs_np, reward_np, terminated, truncated, info = envs.step(act_np)
        next_done = np.logical_or(terminated, truncated)

        ep_returns[:] += reward_np
        final_infos: list = []
        has_final = False
        for i in range(args.num_envs):
            if next_done[i]:
                final_infos.append({"episode": {"r": ep_returns[i]}})
                ep_returns[i] = 0.0
                has_final = True
            else:
                final_infos.append(None)
        infos: dict = {}
        if has_final:
            infos["final_info"] = final_infos

        obs_rms.update(next_obs_np)
        next_obs_norm = np.clip(obs_rms.normalize(next_obs_np), -10, 10)

        disc_returns[:] = reward_np + args.gamma * disc_returns * (~next_done)
        ret_rms.update(disc_returns)
        reward_norm = np.clip(reward_np / np.sqrt(ret_rms.var + 1e-8), -10, 10)

        return (
            torch.as_tensor(next_obs_norm, dtype=torch.float),
            torch.as_tensor(reward_norm, dtype=torch.float),
            torch.as_tensor(next_done),
            infos,
        )

    ####### Agent #######
    agent = Agent(n_obs, n_act, device=device)
    agent_inference = Agent(n_obs, n_act, device=device)
    agent_inference_p = from_module(agent).data
    agent_inference_p.to_module(agent_inference)

    ####### Optimizer #######
    optimizer = optim.Adam(
        agent.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,
        capturable=args.cudagraphs and not args.compile,
    )

    ####### Executables #######
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value

    if args.compile:
        policy = torch.compile(policy)
        gae = torch.compile(gae, fullgraph=True)
        update = torch.compile(update)

    if args.cudagraphs:
        policy = CudaGraphModule(policy)
        gae = CudaGraphModule(gae)
        update = CudaGraphModule(update)

    avg_returns = deque(maxlen=20)
    global_step = 0
    container_local = None

    raw_obs, _ = envs.reset(seed=args.seed)
    obs_rms.update(raw_obs)
    next_obs = torch.tensor(
        np.clip(obs_rms.normalize(raw_obs), -10, 10),
        device=device,
        dtype=torch.float,
    )
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)

    # ── Determine which iterations to record video ──
    if args.capture_video and args.video_interval > 0:
        video_iters = set(range(1, args.num_iterations + 1, args.video_interval))
        video_iters.add(args.num_iterations)  # always record the last one
    else:
        video_iters = set()

    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))
    global_step_burnin = None
    start_time = time.time()

    for iteration in pbar:
        if iteration == args.measure_burnin:
            global_step_burnin = global_step
            start_time = time.time()

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        torch.compiler.cudagraph_mark_step_begin()
        next_obs, next_done, container = rollout(
            next_obs, next_done, avg_returns=avg_returns
        )
        global_step += container.numel()

        container = gae(next_obs, next_done, container)
        container_flat = container.view(-1)

        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(
                args.minibatch_size
            )
            for b in b_inds:
                container_local = container_flat[b]
                out = update(container_local, tensordict_out=tensordict.TensorDict())
                if args.target_kl is not None and out["approx_kl"] > args.target_kl:
                    break
            else:
                continue
            break

        # ── Log EVERY iteration ──
        speed = (
            (global_step - global_step_burnin) / (time.time() - start_time)
            if global_step_burnin is not None
            else 0.0
        )
        r = container["rewards"].mean()
        r_max = container["rewards"].max()
        avg_returns_t = (
            torch.tensor(avg_returns).mean() if len(avg_returns) else torch.tensor(0.0)
        )

        with torch.no_grad():
            log_dict = {
                "charts/episode_return": avg_returns_t.item(),
                "charts/reward_mean": r.item(),
                "charts/reward_max": r_max.item(),
                "charts/learning_rate": float(optimizer.param_groups[0]["lr"]),
                "losses/pg_loss": out["pg_loss"].item(),
                "losses/v_loss": out["v_loss"].item(),
                "losses/entropy": out["entropy_loss"].item(),
                "losses/approx_kl": out["approx_kl"].item(),
                "losses/old_approx_kl": out["old_approx_kl"].item(),
                "losses/clipfrac": out["clipfrac"].item(),
                "losses/grad_norm": out["gn"].item(),
                "rollout/logprobs_mean": container["logprobs"].mean().item(),
                "rollout/advantages_mean": container["advantages"].mean().item(),
                "rollout/returns_mean": container["returns"].mean().item(),
                "rollout/vals_mean": container["vals"].mean().item(),
                "perf/speed_sps": speed,
                "perf/iteration": iteration,
                "perf/global_step": global_step,
            }

        # ── Record eval video ──
        if iteration in video_iters:
            print(f"\n[iter {iteration}] Recording evaluation video...")
            vid, eval_reward = record_eval_video(
                yaml=args.yaml,
                agent_inference=agent_inference,
                obs_rms=obs_rms,
                device=device,
                max_steps=args.video_max_steps,
            )
            if vid is not None:
                log_dict["media/eval_video"] = vid
                print(
                    f"[iter {iteration}] Video captured, eval_return={eval_reward:.2f}"
                )
            else:
                print(f"[iter {iteration}] WARNING: no frames captured!")
            log_dict["charts/eval_return"] = eval_reward

        wandb.log(log_dict, step=global_step)

        lr = optimizer.param_groups[0]["lr"]
        pbar.set_description(
            f"it {iteration}/{args.num_iterations} | "
            f"sps: {speed:,.0f} | "
            f"r_avg: {r:.2f} | "
            f"r_max: {r_max:.2f} | "
            f"ep_ret: {avg_returns_t:.2f} | "
            f"lr: {lr:.2e}"
        )

    total_time = time.time() - start_time
    print(f"\nTraining complete: {global_step:,} steps in {total_time:.1f}s")
    print(f"Average speed: {global_step / total_time:,.0f} SPS")

    envs.close()
    wandb.finish()
