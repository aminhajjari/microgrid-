"""
hma_drl.py
==========
Hierarchical Multi-Agent DRL (HMA-DRL) framework.

Architecture (Section 2.3 of the paper):
  Lower layer  → 4 local agents  (BESS:TD3, EV:TD3, Load:SAC, Grid:SAC)
  Upper layer  → Supervisor agent (SAC) that outputs coordination weights ωi  (Eq. 14)

Global reward shaping:  R(t) = Σ ωi · ri(t)   ← Eq. 14
Reward normalisation:   all signals clipped to [-1, +1]  ← Section 2.4.7
"""

from __future__ import annotations

import numpy as np
import torch

from agents import TD3Agent, SACAgent


# ---------------------------------------------------------------------------
# Observation / action split helpers
# ---------------------------------------------------------------------------
# Full observation:  [soc_bess, soc_ev, pv, load, tariff, sin_h, cos_h]  (dim=7)
# Full action:       [p_bess, p_ev, p_flex, p_grid]  (dim=4)

OBS_DIM  = 7
ACT_DIM  = 4
N_AGENTS = 4   # BESS, EV, Load, Grid

# Local views: each agent sees a subset of the global state
LOCAL_OBS_IDX = {
    "bess": [0, 2, 3, 4, 5, 6],   # soc_bess, pv, load, tariff, sin, cos
    "ev":   [1, 2, 3, 4, 5, 6],   # soc_ev,   pv, load, tariff, sin, cos
    "load": [2, 3, 4, 5, 6],       # pv, load, tariff, sin, cos
    "grid": [2, 3, 4, 5, 6],       # pv, load, tariff, sin, cos
}
LOCAL_ACT_IDX = {"bess": 0, "ev": 1, "load": 2, "grid": 3}


def local_obs(obs: np.ndarray, agent: str) -> np.ndarray:
    return obs[LOCAL_OBS_IDX[agent]]


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
            "bess": TD3Agent(lo_bess, 1, device=d),  # TD3 for BESS
            "ev":   TD3Agent(lo_ev,   1, device=d),  # TD3 for EV
            "load": SACAgent(lo_load, 1, device=d),  # SAC for flexible load
            "grid": SACAgent(lo_grid, 1, device=d),  # SAC for grid
        }

        # --- Supervisor agent (Section 2.3.3) ---
        # Input:  global obs (7) + 4 local rewards = 11 dims
        # Output: 4 importance weights ωi → softmax'd to sum=1
        self.supervisor = SACAgent(OBS_DIM + N_AGENTS, N_AGENTS, device=d)

        # FIX 1: always initialised — avoids getattr bug on first call
        self._last_omega = _softmax(np.zeros(N_AGENTS))
        self._device = torch.device(d)

    # ------------------------------------------------------------------
    def select_actions(
        self,
        obs: np.ndarray,
        local_rewards: np.ndarray | None = None,
        explore: bool = True,
    ) -> np.ndarray:
        """
        Return joint action [p_bess, p_ev, p_flex, p_grid] in [-1, 1].
        local_rewards: last step's per-agent rewards; zeros on first call.
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
        omega     = _softmax(omega_raw)          # ensure ωi > 0 and sum to 1
        self._last_omega = omega.copy()

        # Modulate each local action by its weight (±30 % around neutral 0.25)

        bias_scale = 0.2
        for i, name in enumerate(["bess", "ev", "load", "grid"]):
          idx = LOCAL_ACT_IDX[name]
          action[idx] = np.clip(action[idx] + bias_scale * (omega[i] - 0.25), -1, 1)
          
         return action 

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
        # Reward = weighted sum of local rewards scaled to match env reward range
        sup_obs      = np.concatenate([obs,      prev_local_rewards])
        sup_next_obs = np.concatenate([next_obs, local_rewards])
        sup_action   = omega
        sup_reward   = float(np.dot(omega, local_rewards)) * 10.0  # scale to match env
        self.supervisor.buffer.store(
            sup_obs, sup_action, sup_reward, sup_next_obs, float(done)
        )

    # ------------------------------------------------------------------
    def update_all(self, batch_size: int = 256) -> dict:
        """Update all local agents + supervisor; return loss dict."""
        losses = {}
        for name, agent in self.agents.items():
            info = agent.update(batch_size)
            if info:
                losses[name] = info

        # FIX 4: supervisor trains at same threshold as local agents
        if self.supervisor.buffer.size >= batch_size:
            sup_info = self.supervisor.update(batch_size)
            if sup_info:
                losses["supervisor"] = sup_info

        return losses

    # ------------------------------------------------------------------
    def compute_local_rewards(self, info: dict) -> np.ndarray:
        """
        Decompose the single-step info dict into per-agent reward signals.
        Agents: [BESS, EV, Load, Grid]

        Signs follow the paper:
          BESS: discharge (p_bess > 0) saves grid import  → positive reward
                charge    (p_bess < 0) costs energy       → negative reward
                degradation always subtracts              (Eq. 11)
          EV:   same logic with 0.5 weighting for V2G
          Load: comfort penalty + tariff savings          (Eq. 13)
          Grid: import cost penalty + load-loss penalty   (Eq. 10)

        Normalisation: scale to [-1,+1] per Section 2.4.7,
        using max(|r|, 1) so small rewards are not amplified.
        """
        lam      = info.get("tariff",    0.1)
        p_bess   = info.get("p_bess",    0.0)
        p_ev     = info.get("p_ev",      0.0)
        p_flex   = info.get("p_flex",    info.get("p_load", 30.0))
        p_grid   = info.get("p_grid",    0.0)
        ll       = info.get("load_loss", 0.0)

        from microgrid_env import MicrogridEnv as _E  # import constants

        # FIX B: discharge rewarded, charge penalised; degradation always negative
        r_bess = (lam * p_bess * _E.DT) \
                 - _E.GAMMA * (abs(p_bess) / _E.P_BESS_MAX) ** _E.KAPPA

        # FIX C: V2G discharge rewarded, charging penalised
        r_ev = (lam * p_ev * _E.DT) * 0.5

        # Comfort + tariff savings  (Eq. 13)
        r_load = -abs(p_flex - 30.0) * _E.ZETA \
                 - lam * max(0, p_flex - 30.0) * 0.2

        # Grid import cost + softer load-loss penalty  (FIX 5)
        r_grid = -(lam * max(0, p_grid) * _E.DT) - ll * 0.5

        rewards = np.array([r_bess, r_ev, r_load, r_grid], dtype=np.float32)

        # FIX 2 + Section 2.4.7: normalise to [-1,+1] without collapsing scale
        scale = max(float(np.abs(rewards).max()), 1.0)
        return np.clip(rewards / scale, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Flat Multi-Agent DRL baseline (Section 3.3.1b)
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
# Single-Agent DRL baseline (Section 3.3.1a)
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
# Helper
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x -= x.max()
    e  = np.exp(x)
    return e / e.sum()
