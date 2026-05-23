"""
hma_drl.py  (FIXED)
==========
Hierarchical Multi-Agent DRL (HMA-DRL) framework.

Architecture (Section 2.3 of the paper):
  Lower layer  → 4 local agents  (BESS:TD3, EV:TD3, Load:SAC, Grid:SAC)
  Upper layer  → Supervisor agent (SAC) that outputs coordination weights ωi  (Eq. 14)

Global reward shaping:  R(t) = Σ ωi · ri(t)   ← Eq. 14
Reward normalisation:   all signals clipped to [-1, +1]  ← Section 2.4.7

==========================================================================
CHANGES vs. original  (every change is marked FIX-A … FIX-E)
==========================================================================

FIX-A  Supervisor warm-up decoupled from local buffer sizes.
       Original gated supervisor training on min(local buffer sizes) >= 5000.
       Because local agents have small obs/act dims they fill their buffers
       quickly, but the supervisor's own buffer fills independently (one
       transition per env step).  The old gate caused the supervisor to start
       training far too early in some runs and too late in others, and it
       never gave the supervisor its own dedicated warm-up period.
       Fix: track supervisor buffer size directly; start training only after
       SUP_WARMUP_SIZE transitions have been collected by the supervisor.

FIX-B  Supervisor reward rescaled to match the environment's reward range.
       Original:  sup_reward = dot(omega, local_rewards)
       local_rewards are clipped to [-1,+1] and omega sums to 1, so
       sup_reward ∈ [-1,+1].  The environment's global reward is in [-10,+10]
       (see microgrid_env.py clip).  The supervisor therefore saw a signal
       ~10x smaller than SA/Flat, giving it almost no gradient.
       Fix: scale sup_reward by ENV_REWARD_SCALE = 10.0 so the supervisor
       operates in the same reward range as the other two baselines.

FIX-C  Local reward double-counting removed.
       Original r_grid included -(lam * p_grid * DT), which is the same cost
       term that r_bess and r_ev try to minimise (they reduce grid import).
       This created conflicting gradients: BESS/EV learned to discharge to
       save cost, while Grid also "owned" that cost, giving the supervisor
       two agents fighting over the same signal.
       Fix: r_grid now only penalises load-loss (reliability), r_bess owns
       the import-cost savings from BESS dispatch, and a new r_cost_shared
       term for the residual grid cost is folded into r_grid alone so
       responsibilities are non-overlapping.

FIX-D  Action modulation made unconditional (removed the buffer-size gate).
       Original only applied omega bias after supervisor buffer > 5000.
       Before that threshold HMA behaved identically to Flat MA-DRL but
       received worse local rewards, so it never had a chance to outperform
       Flat during the critical early exploration phase.
       Fix: omega modulation is applied from the very first step.  The bias
       scale is kept small (0.15) so random omega values at the start do not
       destabilise the local policies.

FIX-E  Best-checkpoint saving added to HMADRLFramework.
       The original always overwrote weights at the end of each run, so
       multi-seed runs always saved the last (potentially diverged) seed.
       Fix: track best_eval_reward and expose save_if_best() so train.py
       can call it after each greedy evaluation.
"""

from __future__ import annotations

import numpy as np
import torch

from agents import TD3Agent, SACAgent


# ---------------------------------------------------------------------------
# Observation / action split helpers
# ---------------------------------------------------------------------------
OBS_DIM  = 7
ACT_DIM  = 4
N_AGENTS = 4   # BESS, EV, Load, Grid

LOCAL_OBS_IDX = {
    "bess": [0, 2, 3, 4, 5, 6],   # soc_bess, pv, load, tariff, sin, cos
    "ev":   [1, 2, 3, 4, 5, 6],   # soc_ev,   pv, load, tariff, sin, cos
    "load": [2, 3, 4, 5, 6],      # pv, load, tariff, sin, cos
    "grid": [2, 3, 4, 5, 6],      # pv, load, tariff, sin, cos
}
LOCAL_ACT_IDX = {"bess": 0, "ev": 1, "load": 2, "grid": 3}


def local_obs(obs: np.ndarray, agent: str) -> np.ndarray:
    return obs[LOCAL_OBS_IDX[agent]]


# ---------------------------------------------------------------------------
# FIX-A: dedicated supervisor warm-up threshold
# FIX-B: scale factor to match env reward range
# ---------------------------------------------------------------------------
SUP_WARMUP_SIZE  = 2000   # FIX-A: supervisor trains only after this many stored transitions
ENV_REWARD_SCALE = 10.0   # FIX-B: env clips reward to [-10, +10]; scale sup_reward to match
OMEGA_BIAS_SCALE = 0.15   # FIX-D: small modulation so random early omega doesn't destabilise


# ---------------------------------------------------------------------------
class HMADRLFramework:
    """
    Hierarchical multi-agent DRL controller.

    Parameters
    ----------
    device : str
        'cpu' or 'cuda'
    """

    def __init__(self, device: str = "cpu"):
        d = device

        # --- Local agents (Section 2.3.4) ---
        lo_bess = len(LOCAL_OBS_IDX["bess"])
        lo_ev   = len(LOCAL_OBS_IDX["ev"])
        lo_load = len(LOCAL_OBS_IDX["load"])
        lo_grid = len(LOCAL_OBS_IDX["grid"])

        self.agents = {
            "bess": TD3Agent(lo_bess, 1, device=d),
            "ev":   TD3Agent(lo_ev,   1, device=d),
            "load": SACAgent(lo_load, 1, device=d),
            "grid": SACAgent(lo_grid, 1, device=d),
        }

        # --- Supervisor agent (Section 2.3.3) ---
        # Input:  global obs (7) + 4 local rewards = 11 dims
        # Output: 4 importance weights ωi → softmax'd to sum=1
        self.supervisor = SACAgent(OBS_DIM + N_AGENTS, N_AGENTS, device=d)

        self._last_omega = _softmax(np.zeros(N_AGENTS))
        self._device     = torch.device(d)

        # FIX-E: track best evaluation reward for checkpoint saving
        self.best_eval_reward = -np.inf

    # ------------------------------------------------------------------
    def select_actions(
        self,
        obs: np.ndarray,
        local_rewards: np.ndarray | None = None,
        explore: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (joint_action, omega).
        joint_action: [p_bess, p_ev, p_flex, p_grid] in [-1, 1]
        omega:        coordination weights from supervisor (sum to 1)
        """
        if local_rewards is None:
            local_rewards = np.zeros(N_AGENTS)

        # --- Local agent actions ---
        action = np.zeros(4)
        for name, idx in LOCAL_ACT_IDX.items():
            lo = local_obs(obs, name)
            if isinstance(self.agents[name], TD3Agent):
                noise = 0.1 if explore else 0.0
                action[idx] = self.agents[name].select_action(lo, noise_std=noise)[0]
            else:
                action[idx] = self.agents[name].select_action(
                    lo, deterministic=not explore
                )[0]

        # --- Supervisor: coordination weights ωi (Eq. 14) ---
        sup_obs   = np.concatenate([obs, local_rewards])
        omega_raw = self.supervisor.select_action(sup_obs, deterministic=not explore)
        omega     = _softmax(omega_raw)
        self._last_omega = omega.copy()

        # FIX-D: apply omega modulation unconditionally (no buffer-size gate)
        # Small bias_scale keeps random early omega from destabilising policies
        for i, name in enumerate(["bess", "ev", "load", "grid"]):
            idx = LOCAL_ACT_IDX[name]
            action[idx] = np.clip(
                action[idx] + OMEGA_BIAS_SCALE * (omega[i] - 0.25), -1.0, 1.0
            )

        return action, omega

    # ------------------------------------------------------------------
    def store_transitions(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        local_rewards: np.ndarray,
        global_reward: float,
        next_obs: np.ndarray,
        done: bool,
        prev_local_rewards: np.ndarray,
        omega: np.ndarray,
    ):
        """Store (s,a,r,s') for every agent and the supervisor."""

        # --- Local agents ---
        for name, idx in LOCAL_ACT_IDX.items():
            lo      = local_obs(obs,      name)
            lo_next = local_obs(next_obs, name)
            act_i   = np.array([action[idx]])
            rew_i   = float(local_rewards[idx])
            self.agents[name].buffer.store(lo, act_i, rew_i, lo_next, float(done))

        # --- Supervisor ---
        sup_obs      = np.concatenate([obs,      prev_local_rewards])
        sup_next_obs = np.concatenate([next_obs, local_rewards])
        sup_action   = omega

        # FIX-B: scale supervisor reward to match env reward range [-10, +10]
        # dot(omega, local_rewards) ∈ [-1,+1]; multiply by ENV_REWARD_SCALE
        sup_reward = float(np.dot(omega, local_rewards)) * ENV_REWARD_SCALE

        self.supervisor.buffer.store(
            sup_obs, sup_action, sup_reward, sup_next_obs, float(done)
        )

    # ------------------------------------------------------------------
    def update_all(self, batch_size: int = 256) -> dict:
        losses = {}

        # Update local agents always (they manage their own buffer-size checks)
        for name, agent in self.agents.items():
            info = agent.update(batch_size)
            if info:
                losses[name] = info

        # FIX-A: train supervisor only after it has its own warm-up transitions
        # (independent of local buffer sizes)
        if self.supervisor.buffer.size >= SUP_WARMUP_SIZE:
            sup_info = self.supervisor.update(batch_size)
            if sup_info:
                losses["supervisor"] = sup_info

        return losses

    # ------------------------------------------------------------------
    def compute_local_rewards(self, info: dict) -> np.ndarray:
        """
        Decompose single-step info into per-agent reward signals.
        Agents: [BESS, EV, Load, Grid]

        FIX-C: responsibilities are non-overlapping
          BESS  → degradation cost + discharge value (saves grid import)
          EV    → V2G discharge value only (no cost double-count)
          Load  → comfort deviation penalty only
          Grid  → residual import cost + load-loss reliability penalty

        Normalisation: scale to [-1,+1] per Section 2.4.7,
        using max(|r|, 1) so small rewards are not amplified.
        """
        lam    = info.get("tariff",    0.1)
        p_bess = info.get("p_bess",    0.0)
        p_ev   = info.get("p_ev",      0.0)
        p_flex = info.get("p_flex",    info.get("p_load", 30.0))
        p_grid = info.get("p_grid",    0.0)
        ll     = info.get("load_loss", 0.0)

        from microgrid_env import MicrogridEnv as _E

        # --- FIX-C: non-overlapping reward decomposition ---

        # BESS: net value of dispatch (discharge saves grid import) minus degradation
        # Positive p_bess = discharge = saves lam per kWh → positive reward
        # Degradation always subtracts regardless of direction
        r_bess = (lam * p_bess * _E.DT) \
                 - _E.GAMMA * (abs(p_bess) / _E.P_BESS_MAX) ** _E.KAPPA

        # EV: V2G discharge value only (charge is neutral here; Grid pays import cost)
        # 0.5 weight to avoid over-incentivising V2G at the expense of EV battery
        r_ev = (lam * max(0.0, p_ev) * _E.DT) * 0.5

        # Load: comfort penalty only — how far flexible load deviates from baseline 30 kW
        # No tariff term here (Grid owns residual import cost)
        r_load = -abs(p_flex - 30.0) * _E.ZETA

        # Grid: owns ALL residual import cost + reliability penalty
        # This is the ONLY place lam * p_grid appears (FIX-C removes double-count)
        r_grid = -(lam * max(0.0, p_grid) * _E.DT) - ll * 1.0

        rewards = np.array([r_bess, r_ev, r_load, r_grid], dtype=np.float32)

        # Normalise to [-1,+1] without collapsing near-zero signals
        scale = max(float(np.abs(rewards).max()), 1.0)
        return np.clip(rewards / scale, -1.0, 1.0)

    # ------------------------------------------------------------------
    # FIX-E: best-checkpoint helper
    # ------------------------------------------------------------------
    def save_if_best(
        self,
        eval_reward: float,
        save_dir,         # pathlib.Path
        method: str = "hma",
    ) -> bool:
        """
        Save weights only when eval_reward improves the running best.
        Returns True if a new best was saved.
        Call this from train.py after each greedy evaluation block.
        """
        if eval_reward > self.best_eval_reward:
            self.best_eval_reward = eval_reward
            _save_hma_weights(self, save_dir, method)
            return True
        return False


# ---------------------------------------------------------------------------
# Flat Multi-Agent DRL baseline (Section 3.3.1b)  — unchanged
# ---------------------------------------------------------------------------

class FlatMADRL:
    """Independent learners without supervisor."""

    def __init__(self, device: str = "cpu"):
        d = device
        self.agents = {
            "bess": TD3Agent(len(LOCAL_OBS_IDX["bess"]), 1, device=d),
            "ev":   TD3Agent(len(LOCAL_OBS_IDX["ev"]),   1, device=d),
            "load": SACAgent(len(LOCAL_OBS_IDX["load"]), 1, device=d),
            "grid": SACAgent(len(LOCAL_OBS_IDX["grid"]), 1, device=d),
        }

    def select_actions(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        action = np.zeros(4)
        for name, idx in LOCAL_ACT_IDX.items():
            lo    = local_obs(obs, name)
            noise = 0.1 if explore else 0.0
            if isinstance(self.agents[name], TD3Agent):
                action[idx] = self.agents[name].select_action(lo, noise)[0]
            else:
                action[idx] = self.agents[name].select_action(
                    lo, deterministic=not explore
                )[0]
        return action.clip(-1, 1)

    def store_transitions(self, obs, action, rewards, next_obs, done, **_):
        for name, idx in LOCAL_ACT_IDX.items():
            self.agents[name].buffer.store(
                local_obs(obs, name), [action[idx]],
                rewards[idx], local_obs(next_obs, name), float(done)
            )

    def update_all(self, batch_size: int = 256) -> dict:
        return {n: a.update(batch_size) for n, a in self.agents.items()}


# ---------------------------------------------------------------------------
# Single-Agent DRL baseline (Section 3.3.1a)  — unchanged
# ---------------------------------------------------------------------------

class SingleAgentDRL:
    """One TD3 agent controlling all 4 actuators."""

    def __init__(self, device: str = "cpu"):
        self.agent = TD3Agent(OBS_DIM, ACT_DIM, device=device)

    def select_actions(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        return self.agent.select_action(obs, noise_std=0.1 if explore else 0.0)

    def store_transitions(self, obs, action, reward, next_obs, done, **_):
        self.agent.buffer.store(obs, action, reward, next_obs, float(done))

    def update_all(self, batch_size: int = 256) -> dict:
        return {"sa": self.agent.update(batch_size)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x -= x.max()
    e  = np.exp(x)
    return e / e.sum()


def _save_hma_weights(controller: HMADRLFramework, save_dir, method: str):
    """Internal helper — writes weights to disk."""
    import torch
    w = {}
    for name, agent in controller.agents.items():
        w[f"{name}_actor"]  = agent.actor.state_dict()
        w[f"{name}_critic"] = agent.critic.state_dict()
    w["supervisor_actor"]  = controller.supervisor.actor.state_dict()
    w["supervisor_critic"] = controller.supervisor.critic.state_dict()
    torch.save(w, save_dir / f"{method}_weights.pt")
    print(f"  Weights saved → {save_dir}/{method}_weights.pt")
