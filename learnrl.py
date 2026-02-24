from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import tyro
import wandb
from torch.distributions.normal import Normal

from rustoracerpy import RustoracerEnv


# ━━━━━━━━━━━━━━━━━━━━  hyper-parameters  ━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 42
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = True
    """whether to capture videos of the agent performances"""

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
    max_ep_steps: int = 10_000
    """maximum steps per episode"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.02
    """the target KL divergence threshold"""

    # Network
    hidden: int = 256
    """hidden layer size"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    video_interval: int = 25
    """record an evaluation video every N iterations (0 to disable)"""
    video_max_steps: int = 600
    """max steps per evaluation video episode"""

    save_interval: int = 50
    """save checkpoint every N iterations"""
    save_dir: str = "checkpoints"
    """directory to save checkpoints"""


# ━━━━━━━━━━━━  running mean / std (Welford, CPU f64)  ━━━━━━━━━━━
class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count: float = 1e-4

    def update(self, batch: np.ndarray) -> None:
        batch = batch.reshape(-1, *self.mean.shape)
        bm, bv, bc = batch.mean(0), batch.var(0), batch.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        new_mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta**2 * self.count * bc / tot
        self.mean, self.var, self.count = new_mean, m2 / tot, tot

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        normed = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normed, -clip, clip)


class RunningReturnStd:
    """Track discounted return variance for reward normalisation."""

    def __init__(self, num_envs: int, gamma: float) -> None:
        self.gamma = gamma
        self.returns = np.zeros(num_envs, np.float64)
        self.rms = RunningMeanStd(shape=())

    def update(self, rewards: np.ndarray, dones: np.ndarray) -> None:
        self.returns = self.returns * self.gamma * (1.0 - dones) + rewards
        self.rms.update(self.returns)

    def normalize(self, rewards: np.ndarray) -> np.ndarray:
        return rewards / np.sqrt(self.rms.var + 1e-8)


# ━━━━━━━━━━━━━━━━━━  actor-critic network  ━━━━━━━━━━━━━━━━━━━━━━
LOG_STD_MIN = -5.0
LOG_STD_MAX = 0.5


def layer_init(
    layer: nn.Linear, std: float = float(np.sqrt(2)), bias: float = 0.0
) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), -0.5))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(x)
        log_std = self.actor_logstd.expand_as(mean).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        return (
            action,
            dist.log_prob(action).sum(-1),
            dist.entropy().sum(-1),
            self.critic(x).squeeze(-1),
        )


def has_nan(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            return True
    return False


# ─────────────────────────────────────────────────────────────
# Video recording helper
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def record_eval_video(
    yaml: str,
    agent: Agent,
    obs_rms: RunningMeanStd,
    device: torch.device,
    max_steps: int = 600,
    max_ep_steps: int = 10_000,
) -> Tuple["wandb.Video | None", float]:
    """Run one deterministic episode, return (wandb.Video, total_reward)."""
    eval_env = RustoracerEnv(
        yaml=yaml,
        num_envs=1,
        max_steps=max_ep_steps,
        render_mode="rgb_array",
    )
    raw_obs, _ = eval_env.reset(seed=42)

    frames: list[np.ndarray] = []
    total_reward = 0.0

    for _ in range(max_steps):
        frame = eval_env.render()
        if frame is not None:
            frames.append(frame)

        obs_norm = obs_rms.normalize(raw_obs)
        obs_t = torch.tensor(obs_norm, device=device, dtype=torch.float32)
        action_mean = agent.actor_mean(obs_t)
        action_np = action_mean.cpu().numpy().astype(np.float64).clip(-1.0, 1.0)

        raw_obs, reward, terminated, truncated, _ = eval_env.step(action_np)
        total_reward += float(reward[0])

        if terminated[0] or truncated[0]:
            break

    eval_env.close()

    if len(frames) == 0:
        return None, total_reward

    video_np = np.stack(frames, axis=0).transpose(0, 3, 1, 2)  # (T,C,H,W)
    video = wandb.Video(video_np, fps=60, format="mp4")
    return video, total_reward


# ─────────────────────────────────────────────────────────────
# Checkpoint helper
# ─────────────────────────────────────────────────────────────
def save_checkpoint(
    args: Args,
    agent: Agent,
    optimizer: optim.Adam,
    obs_rms: RunningMeanStd,
    ret_rms: RunningReturnStd,
    global_step: int,
    update: int,
    name: str | None = None,
) -> None:
    if name is None:
        name = f"agent_{global_step}.pt"
    path = os.path.join(args.save_dir, name)
    torch.save(
        {
            "model": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "obs_rms_mean": obs_rms.mean,
            "obs_rms_var": obs_rms.var,
            "obs_rms_count": obs_rms.count,
            "ret_rms_var": ret_rms.rms.var,
            "ret_rms_count": ret_rms.rms.count,
            "global_step": global_step,
            "update": update,
        },
        path,
    )
    print(f"  💾  {path}")


# ─────────────────────────────────────────────────────────────
# Main training loop (mirrors ppo.py exactly)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = tyro.cli(Args)

    batch_size = args.num_envs * args.num_steps
    args.minibatch_size = batch_size // args.num_minibatches
    args.batch_size = batch_size
    args.num_iterations = args.total_timesteps // batch_size
    run_name = f"Rustoracer__{args.exp_name}__{args.seed}"

    print(f"batch_size        = {args.batch_size:,}")
    print(f"minibatch_size    = {args.minibatch_size:,}")
    print(f"num_iterations    = {args.num_iterations}")
    print(f"hidden            = {args.hidden}")
    print(f"update_epochs     = {args.update_epochs}")
    print(f"ent_coef          = {args.ent_coef}")
    print(f"target_kl         = {args.target_kl}")
    print(f"max_ep_steps      = {args.max_ep_steps}")

    wandb.init(
        project="ppo_continuous_action",
        name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
        config=vars(args),
        save_code=True,
    )

    # ── seeding ──────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # ── device ───────────────────────────────────────────────────
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device : {device}")

    # ── environment ──────────────────────────────────────────────
    env = RustoracerEnv(
        yaml=args.yaml,
        num_envs=args.num_envs,
        max_steps=args.max_ep_steps,
    )
    obs_dim: int = env.single_observation_space.shape[0]
    act_dim: int = env.single_action_space.shape[0]
    print(f"obs={obs_dim}  act={act_dim}  envs={args.num_envs}")

    # ── network + optimiser ──────────────────────────────────────
    agent = Agent(obs_dim, act_dim, args.hidden).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    print(f"Parameters : {sum(p.numel() for p in agent.parameters()):,}")

    # ── normalisers ──────────────────────────────────────────────
    obs_rms = RunningMeanStd(shape=(obs_dim,))
    ret_rms = RunningReturnStd(args.num_envs, args.gamma)

    # ── rollout buffers (CPU, f64) ───────────────────────────────
    obs_buf = np.zeros((args.num_steps, args.num_envs, obs_dim), np.float64)
    raw_act_buf = np.zeros((args.num_steps, args.num_envs, act_dim), np.float64)
    logp_buf = np.zeros((args.num_steps, args.num_envs), np.float64)
    rew_buf = np.zeros((args.num_steps, args.num_envs), np.float64)
    done_buf = np.zeros((args.num_steps, args.num_envs), np.float64)
    val_buf = np.zeros((args.num_steps, args.num_envs), np.float64)

    # ── episode trackers ─────────────────────────────────────────
    ep_ret = np.zeros(args.num_envs, np.float64)
    ep_len = np.zeros(args.num_envs, np.int64)
    recent_returns: list[float] = []
    recent_lengths: list[int] = []

    # ── first reset ──────────────────────────────────────────────
    next_obs_raw, _ = env.reset(seed=args.seed)
    obs_rms.update(next_obs_raw)
    next_obs_n = obs_rms.normalize(next_obs_raw)
    next_done = np.zeros(args.num_envs, np.float64)

    os.makedirs(args.save_dir, exist_ok=True)
    global_step = 0
    t0 = time.time()

    # ── video iteration schedule ─────────────────────────────────
    if args.capture_video and args.video_interval > 0:
        video_iters = set(range(1, args.num_iterations + 1, args.video_interval))
        video_iters.add(args.num_iterations)
    else:
        video_iters = set()

    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))

    # ═══════════════════  outer loop  ════════════════════════════
    for update in pbar:
        # ── learning-rate annealing ──────────────────────────────
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / args.num_iterations
            optimizer.param_groups[0]["lr"] = args.learning_rate * frac

        # ── rollout collection ───────────────────────────────────
        for step in range(args.num_steps):
            global_step += args.num_envs

            obs_buf[step] = next_obs_n
            done_buf[step] = next_done

            with torch.no_grad():
                obs_t = torch.tensor(next_obs_n, dtype=torch.float32, device=device)
                action, logprob, _, value = agent.get_action_and_value(obs_t)
                raw_act_np = action.cpu().numpy()
                logp_buf[step] = logprob.cpu().numpy()
                val_buf[step] = value.cpu().numpy()

            # store RAW action for consistent log-prob ratios
            raw_act_buf[step] = raw_act_np

            # clip only for the environment
            act_clipped = np.clip(raw_act_np, -1.0, 1.0).astype(np.float64)

            next_obs_raw, reward, terminated, truncated, _info = env.step(act_clipped)
            done = np.logical_or(terminated, truncated).astype(np.float64)

            # reward normalisation
            ret_rms.update(reward, done)
            rew_buf[step] = ret_rms.normalize(reward)

            next_done = done

            # episode bookkeeping (on *original* rewards)
            ep_ret += reward
            ep_len += 1
            for i in np.where(done > 0.5)[0]:
                recent_returns.append(float(ep_ret[i]))
                recent_lengths.append(int(ep_len[i]))
                ep_ret[i] = 0.0
                ep_len[i] = 0

            obs_rms.update(next_obs_raw)
            next_obs_n = obs_rms.normalize(next_obs_raw)

        # ── GAE ──────────────────────────────────────────────────
        with torch.no_grad():
            next_val = (
                agent.get_value(
                    torch.tensor(next_obs_n, dtype=torch.float32, device=device)
                )
                .squeeze(-1)
                .cpu()
                .numpy()
            )

        advantages = np.zeros_like(rew_buf)
        lastgae = np.zeros(args.num_envs, np.float64)
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nnt = 1.0 - next_done
                nv = next_val
            else:
                nnt = 1.0 - done_buf[t + 1]
                nv = val_buf[t + 1]
            delta = rew_buf[t] + args.gamma * nv * nnt - val_buf[t]
            lastgae = delta + args.gamma * args.gae_lambda * nnt * lastgae
            advantages[t] = lastgae
        returns = advantages + val_buf

        # ── flatten & move to device ─────────────────────────────
        b_obs = torch.tensor(
            obs_buf.reshape(batch_size, obs_dim), dtype=torch.float32
        ).to(device)
        b_act = torch.tensor(
            raw_act_buf.reshape(batch_size, act_dim), dtype=torch.float32
        ).to(device)
        b_logp = torch.tensor(logp_buf.reshape(batch_size), dtype=torch.float32).to(
            device
        )
        b_adv = torch.tensor(advantages.reshape(batch_size), dtype=torch.float32).to(
            device
        )
        b_ret = torch.tensor(returns.reshape(batch_size), dtype=torch.float32).to(
            device
        )
        b_val = torch.tensor(val_buf.reshape(batch_size), dtype=torch.float32).to(
            device
        )

        # ── PPO update epochs ────────────────────────────────────
        clipfracs: list[float] = []
        kl_early_stopped = False
        pg_loss_val = 0.0
        v_loss_val = 0.0
        ent_loss_val = 0.0
        approx_kl_val = 0.0

        for _epoch in range(args.update_epochs):
            if kl_early_stopped:
                break
            b_inds = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, args.minibatch_size):
                mb = b_inds[start : start + args.minibatch_size]

                _, newlogp, entropy, newval = agent.get_action_and_value(
                    b_obs[mb], b_act[mb]
                )

                # NaN guard on forward pass
                if torch.isnan(newlogp).any() or torch.isnan(newval).any():
                    kl_early_stopped = True
                    break

                logratio = newlogp - b_logp[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                # per-minibatch KL early stop
                if approx_kl > args.target_kl * 1.5:
                    kl_early_stopped = True
                    break

                # advantage normalisation
                mb_adv = b_adv[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # ── policy loss ──
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                # ── value loss ──
                if args.clip_vloss:
                    v_unclipped = (newval - b_ret[mb]) ** 2
                    v_clipped = b_val[mb] + (newval - b_val[mb]).clamp(
                        -args.clip_coef, args.clip_coef
                    )
                    v_loss = (
                        0.5
                        * torch.max(v_unclipped, (v_clipped - b_ret[mb]) ** 2).mean()
                    )
                else:
                    v_loss = 0.5 * ((newval - b_ret[mb]) ** 2).mean()

                ent_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()

                # NaN guard on gradients
                if has_nan(agent):
                    optimizer.zero_grad()
                    kl_early_stopped = True
                    break

                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                pg_loss_val = pg_loss.item()
                v_loss_val = v_loss.item()
                ent_loss_val = ent_loss.item()
                approx_kl_val = approx_kl

        # ── logging ──────────────────────────────────────────────
        elapsed = time.time() - t0
        sps = global_step / elapsed
        mr = np.mean(recent_returns[-200:]) if recent_returns else 0.0
        ml = np.mean(recent_lengths[-200:]) if recent_lengths else 0.0

        log_dict = {
            "charts/episode_return": mr,
            "charts/episode_length": ml,
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "charts/SPS": sps,
            "losses/pg_loss": pg_loss_val,
            "losses/v_loss": v_loss_val,
            "losses/entropy": ent_loss_val,
            "losses/approx_kl": approx_kl_val,
            "losses/clipfrac": np.mean(clipfracs) if clipfracs else 0.0,
            "debug/kl_early_stopped": int(kl_early_stopped),
            "perf/global_step": global_step,
        }

        # ── video ────────────────────────────────────────────────
        if update in video_iters:
            print(f"\n[iter {update}] Recording evaluation video...")
            agent.eval()
            vid, eval_reward = record_eval_video(
                yaml=args.yaml,
                agent=agent,
                obs_rms=obs_rms,
                device=device,
                max_steps=args.video_max_steps,
                max_ep_steps=args.max_ep_steps,
            )
            agent.train()
            if vid is not None:
                log_dict["media/eval_video"] = vid
                print(f"[iter {update}] Video captured, eval_return={eval_reward:.2f}")
            else:
                print(f"[iter {update}] WARNING: no frames captured!")
            log_dict["charts/eval_return"] = eval_reward

        wandb.log(log_dict, step=global_step)

        pbar.set_description(
            f"upd {update}/{args.num_iterations} | "
            f"SPS {sps:,.0f} | "
            f"ret {mr:.2f} | "
            f"len {ml:.0f} | "
            f"pg {pg_loss_val:.4f} | "
            f"vl {v_loss_val:.4f} | "
            f"ent {ent_loss_val:.3f} | "
            f"kl {approx_kl_val:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.1e}"
            f"{'  ⚠KL-STOP' if kl_early_stopped else ''}"
        )

        # ── checkpoint ───────────────────────────────────────────
        if update % args.save_interval == 0:
            save_checkpoint(
                args, agent, optimizer, obs_rms, ret_rms, global_step, update
            )

    # ── final save ───────────────────────────────────────────────
    save_checkpoint(
        args,
        agent,
        optimizer,
        obs_rms,
        ret_rms,
        global_step,
        args.num_iterations,
        name="agent_final.pt",
    )
    total_time = time.time() - t0
    print(f"\n✅  Training complete — {global_step:,} total steps in {total_time:.1f}s")
    print(f"Average speed: {global_step / total_time:,.0f} SPS")

    env.close()
    wandb.finish()
