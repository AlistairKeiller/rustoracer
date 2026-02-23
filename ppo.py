#!/usr/bin/env python3
"""
PPO training for the Rustoracer F1Tenth environment.
Optimised for Apple M2 Pro (MPS backend) + 1024 parallel Rust envs.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from rustoracerpy import RustoracerEnv


# ━━━━━━━━━━━━━━━━━━━━  hyper-parameters  ━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class Config:
    # environment
    map_yaml: str = "maps/berlin.yaml"
    num_envs: int = 1024
    max_ep_steps: int = 10_000

    # PPO core
    total_timesteps: int = 50_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_steps: int = 128
    num_minibatches: int = 32
    update_epochs: int = 4
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    anneal_lr: bool = True
    target_kl: float = 0.02

    # network
    hidden: int = 256

    # system
    device: str = "mps"
    seed: int = 42
    log_interval: int = 1
    save_interval: int = 50
    save_dir: str = "checkpoints"


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
        # start with std ≈ 0.6 so most samples land in [-1, 1]
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


# ━━━━━━━━━━━━━━━━━━━━━━  NaN guard  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def has_nan(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            return True
    return False


# ━━━━━━━━━━━━━━━━━━━━  training loop  ━━━━━━━━━━━━━━━━━━━━━━━━━━
def train(cfg: Config) -> None:
    # ── seeding ──────────────────────────────────────────────────
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ── device ───────────────────────────────────────────────────
    if cfg.device == "mps" and not torch.backends.mps.is_available():
        print("⚠  MPS unavailable → falling back to CPU")
        cfg.device = "cpu"
    device = torch.device(cfg.device)
    print(f"Device : {device}")

    # ── environment ──────────────────────────────────────────────
    env = RustoracerEnv(
        yaml=cfg.map_yaml,
        num_envs=cfg.num_envs,
        max_steps=cfg.max_ep_steps,
    )
    obs_dim: int = env.single_observation_space.shape[0]  # 110
    act_dim: int = env.single_action_space.shape[0]  # 2
    print(f"obs={obs_dim}  act={act_dim}  envs={cfg.num_envs}")

    # ── network + optimiser ──────────────────────────────────────
    agent = Agent(obs_dim, act_dim, cfg.hidden).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
    print(f"Parameters : {sum(p.numel() for p in agent.parameters()):,}")

    # ── derived sizes ────────────────────────────────────────────
    batch_size = cfg.num_envs * cfg.num_steps
    minibatch_size = batch_size // cfg.num_minibatches
    num_updates = cfg.total_timesteps // batch_size
    print(f"batch={batch_size:,}  mini={minibatch_size:,}  updates={num_updates:,}")

    # ── normalisers ──────────────────────────────────────────────
    obs_rms = RunningMeanStd(shape=(obs_dim,))
    ret_rms = RunningReturnStd(cfg.num_envs, cfg.gamma)

    # ── rollout buffers (CPU, f64) ───────────────────────────────
    obs_buf = np.zeros((cfg.num_steps, cfg.num_envs, obs_dim), np.float64)
    raw_act_buf = np.zeros((cfg.num_steps, cfg.num_envs, act_dim), np.float64)
    logp_buf = np.zeros((cfg.num_steps, cfg.num_envs), np.float64)
    rew_buf = np.zeros((cfg.num_steps, cfg.num_envs), np.float64)
    done_buf = np.zeros((cfg.num_steps, cfg.num_envs), np.float64)
    val_buf = np.zeros((cfg.num_steps, cfg.num_envs), np.float64)

    # ── episode trackers ─────────────────────────────────────────
    ep_ret = np.zeros(cfg.num_envs, np.float64)
    ep_len = np.zeros(cfg.num_envs, np.int64)
    recent_returns: list[float] = []
    recent_lengths: list[int] = []

    # ── first reset ──────────────────────────────────────────────
    next_obs_raw, _ = env.reset(seed=cfg.seed)
    obs_rms.update(next_obs_raw)
    next_obs_n = obs_rms.normalize(next_obs_raw)
    next_done = np.zeros(cfg.num_envs, np.float64)

    os.makedirs(cfg.save_dir, exist_ok=True)
    global_step = 0
    t0 = time.time()

    # ═══════════════════  outer loop  ════════════════════════════
    for update in range(1, num_updates + 1):
        # ── learning-rate annealing ──────────────────────────────
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            optimizer.param_groups[0]["lr"] = cfg.learning_rate * frac

        # ── rollout collection ───────────────────────────────────
        for step in range(cfg.num_steps):
            global_step += cfg.num_envs

            obs_buf[step] = next_obs_n
            done_buf[step] = next_done

            with torch.no_grad():
                obs_t = torch.tensor(next_obs_n, dtype=torch.float32, device=device)
                action, logprob, _, value = agent.get_action_and_value(obs_t)
                raw_act_np = action.cpu().numpy()  # unclipped!
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
        lastgae = np.zeros(cfg.num_envs, np.float64)
        for t in reversed(range(cfg.num_steps)):
            if t == cfg.num_steps - 1:
                nnt = 1.0 - next_done
                nv = next_val
            else:
                nnt = 1.0 - done_buf[t + 1]
                nv = val_buf[t + 1]
            delta = rew_buf[t] + cfg.gamma * nv * nnt - val_buf[t]
            lastgae = delta + cfg.gamma * cfg.gae_lambda * nnt * lastgae
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

        for _epoch in range(cfg.update_epochs):
            if kl_early_stopped:
                break
            b_inds = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                mb = b_inds[start : start + minibatch_size]

                _, newlogp, entropy, newval = agent.get_action_and_value(
                    b_obs[mb], b_act[mb]
                )

                # check for NaN in forward pass
                if torch.isnan(newlogp).any() or torch.isnan(newval).any():
                    kl_early_stopped = True
                    break

                logratio = newlogp - b_logp[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    )

                # per-minibatch KL early stop
                if approx_kl > cfg.target_kl * 1.5:
                    kl_early_stopped = True
                    break

                # advantage normalisation
                mb_adv = b_adv[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # ── policy loss ──
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                # ── value loss ──
                if cfg.clip_vloss:
                    v_unclipped = (newval - b_ret[mb]) ** 2
                    v_clipped = b_val[mb] + (newval - b_val[mb]).clamp(
                        -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss = (
                        0.5
                        * torch.max(v_unclipped, (v_clipped - b_ret[mb]) ** 2).mean()
                    )
                else:
                    v_loss = 0.5 * ((newval - b_ret[mb]) ** 2).mean()

                ent_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()

                # NaN guard on gradients
                if has_nan(agent):
                    optimizer.zero_grad()
                    kl_early_stopped = True
                    break

                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

                pg_loss_val = pg_loss.item()
                v_loss_val = v_loss.item()
                ent_loss_val = ent_loss.item()
                approx_kl_val = approx_kl

        # ── logging ──────────────────────────────────────────────
        if update % cfg.log_interval == 0:
            elapsed = time.time() - t0
            sps = global_step / elapsed
            mr = np.mean(recent_returns[-200:]) if recent_returns else 0.0
            ml = np.mean(recent_lengths[-200:]) if recent_lengths else 0.0
            print(
                f"upd {update:>5}/{num_updates} │ "
                f"step {global_step:>11,} │ "
                f"SPS {sps:>9,.0f} │ "
                f"ret {mr:>9.2f} │ "
                f"len {ml:>7.0f} │ "
                f"pg {pg_loss_val:>7.4f} │ "
                f"vl {v_loss_val:>7.4f} │ "
                f"ent {ent_loss_val:>6.3f} │ "
                f"kl {approx_kl_val:>.4f} │ "
                f"clip {np.mean(clipfracs) if clipfracs else 0:>.3f} │ "
                f"lr {optimizer.param_groups[0]['lr']:.1e}"
                f"{'  ⚠KL-STOP' if kl_early_stopped else ''}"
            )

        # ── checkpoint ───────────────────────────────────────────
        if update % cfg.save_interval == 0:
            _save(cfg, agent, optimizer, obs_rms, ret_rms, global_step, update)

    # ── final save ───────────────────────────────────────────────
    _save(
        cfg,
        agent,
        optimizer,
        obs_rms,
        ret_rms,
        global_step,
        num_updates,
        name="agent_final.pt",
    )
    print(f"\n✅  Training complete — {global_step:,} total steps")
    env.close()


def _save(cfg, agent, optimizer, obs_rms, ret_rms, step, update, name=None):
    if name is None:
        name = f"agent_{step}.pt"
    path = os.path.join(cfg.save_dir, name)
    torch.save(
        {
            "model": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "obs_rms_mean": obs_rms.mean,
            "obs_rms_var": obs_rms.var,
            "obs_rms_count": obs_rms.count,
            "ret_rms_var": ret_rms.rms.var,
            "ret_rms_count": ret_rms.rms.count,
            "global_step": step,
            "update": update,
        },
        path,
    )
    print(f"  💾  {path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━  evaluation  ━━━━━━━━━━━━━━━━━━━━━━━━━━
@torch.no_grad()
def evaluate(
    checkpoint: str,
    yaml: str = "maps/berlin.yaml",
    episodes: int = 20,
    device: str = "mps",
) -> None:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    obs_dim, act_dim = 110, 2
    agent = Agent(obs_dim, act_dim).to(device)
    agent.load_state_dict(ckpt["model"])
    agent.eval()

    obs_rms = RunningMeanStd(shape=(obs_dim,))
    obs_rms.mean = ckpt["obs_rms_mean"]
    obs_rms.var = ckpt["obs_rms_var"]
    obs_rms.count = ckpt["obs_rms_count"]

    env = RustoracerEnv(yaml=yaml, num_envs=1, max_steps=10_000)
    returns: list[float] = []

    while len(returns) < episodes:
        obs, _ = env.reset()
        obs = obs_rms.normalize(obs)
        done, total = False, 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            act = agent.actor_mean(obs_t).cpu().numpy()
            act = np.clip(act, -1.0, 1.0).astype(np.float64)
            obs, rew, term, trunc, _ = env.step(act)
            obs = obs_rms.normalize(obs)
            done = bool(term[0] or trunc[0])
            total += float(rew[0])
        returns.append(total)
        print(f"  episode {len(returns):>3}  return {total:>9.2f}")

    print(f"\nmean={np.mean(returns):.2f}  std={np.std(returns):.2f}")
    env.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━  CLI  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    tr = sub.add_parser("train", help="Run PPO training")
    tr.add_argument("--map", default="maps/berlin.yaml")
    tr.add_argument("--num-envs", type=int, default=1024)
    tr.add_argument("--total-timesteps", type=int, default=50_000_000)
    tr.add_argument("--num-steps", type=int, default=128)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--gamma", type=float, default=0.99)
    tr.add_argument("--hidden", type=int, default=256)
    tr.add_argument("--device", default="mps")
    tr.add_argument("--seed", type=int, default=42)

    ev = sub.add_parser("eval", help="Evaluate a checkpoint")
    ev.add_argument("checkpoint")
    ev.add_argument("--map", default="maps/berlin.yaml")
    ev.add_argument("--episodes", type=int, default=20)
    ev.add_argument("--device", default="mps")

    args = p.parse_args()

    if args.cmd == "eval":
        evaluate(args.checkpoint, args.map, args.episodes, args.device)
    else:
        cfg = Config(
            map_yaml=getattr(args, "map", "maps/berlin.yaml"),
            num_envs=getattr(args, "num_envs", 1024),
            total_timesteps=getattr(args, "total_timesteps", 50_000_000),
            num_steps=getattr(args, "num_steps", 128),
            learning_rate=getattr(args, "lr", 3e-4),
            gamma=getattr(args, "gamma", 0.99),
            hidden=getattr(args, "hidden", 256),
            device=getattr(args, "device", "mps"),
            seed=getattr(args, "seed", 42),
        )
        train(cfg)
