"""
hma_drl.py  (FIXED v2)
==========
Hierarchical Multi-Agent DRL (HMA-DRL) framework.

==========================================================================
CHANGES in v2  (on top of the FIX-A…FIX-E from the previous round)
==========================================================================

FIX-F  Omega modulation restricted to storage agents only (BESS + EV).
       v1 applied the omega bias to ALL four actions including Load and Grid.
       Grid is safety-critical: if random omega reduces p_grid before the
       local grid agent has learned to compensate, the microgrid cannot
       meet load demand, causing load-loss events.  This is exactly what
       the new run showed: HMA LOLP jumped from 7.7% to 17.9% while SA
       and Flat stayed at 7-8%.  The fix: only BESS and EV actions are
       modulated by omega.  Load and Grid agents act purely on their own
       learned policies — the supervisor steers energy storage, not
       safety-critical supply.

       Before (v1):
           for i, name in enumerate(["bess", "ev", "load", "grid"]):
               action[idx] += OMEGA_BIAS_SCALE * (omega[i] - 0.25)

       After (v2):
           for i, name in enumerate(["bess", "ev"]):   # storage only
               action[idx] += OMEGA_BIAS_SCALE * (omega[i] - 0.25)
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
    "bess": [0, 2, 3, 4, 5, 6],
    "ev":   [1, 2, 3, 4, 5, 6],
    "load": [2, 3, 4, 5, 6],
    "grid": [2, 3, 4, 5, 6],
}
LOCAL_ACT_IDX = {"bess": 0, "ev": 1, "load": 2, "grid": 3}


def local_obs(obs: np.ndarray, agent: str) -> np.ndarray:
    return obs[LOCAL_OBS_IDX[agent]]


SUP_WARMUP_SIZE  = 2000   # FIX-A: supervisor trains after this many own transitions
ENV_REWARD_SCALE = 10.0   # FIX-B: scale sup_reward to match env [-10,+10] range
OMEGA_BIAS_SCALE = 0.15   # FIX-D/F: modulation scale for storage agents only

# FIX-F: only these agents get omega modulation (storage; safety-critical excluded)
OMEGA_MODULATED_AGENTS = ["bess", "ev"]


class HMADRLFramework:
    """Hierarchical multi-agent DRL controller."""

    def __init__(self, device: str = "cpu"):
        d = device

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

        # Supervisor: input = global obs (7) + 4 local rewards = 11 dims
        self.supervisor = SACAgent(OBS_DIM + N_AGENTS, N_AGENTS, device=d)

        self._last_omega      = _softmax(np.zeros(N_AGENTS))
        self._device          = torch.device(d)
        self.best_eval_reward = -np.inf   # FIX-E

    # ------------------------------------------------------------------
    def select_actions(
        self,
        obs: np.ndarray,
        local_rewards: np.ndarray | None = None,
        explore: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if local_rewards is None:
            local_rewards = np.zeros(N_AGENTS)

        # Local agent actions
        action = np.zeros(4)
        for name, idx in LOCAL_ACT_IDX.items():
            lo = local_obs(obs, name)
            if isinstance(self.agents[name], TD3Agent):
                noise = 0.1 if explore else 0.0
                action[idx] = self.agents[name].select_action(lo, noise_std=noise)[0]
            else:
                action[idx] = self.agents[name].select_action(
                    lo, deterministic=not explore)[0]

        # Supervisor coordination weights
        sup_obs   = np.concatenate([obs, local_rewards])
        omega_raw = self.supervisor.select_action(sup_obs, deterministic=not explore)
        omega     = _softmax(omega_raw)
        self._last_omega = omega.copy()

        # FIX-F: only modulate storage agents (BESS + EV), NOT Load or Grid
        # Load/Grid are safety-critical supply paths — random omega from an
        # untrained supervisor must not restrict them
        for i, name in enumerate(OMEGA_MODULATED_AGENTS):
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
        # Local agents
        for name, idx in LOCAL_ACT_IDX.items():
            lo      = local_obs(obs,      name)
            lo_next = local_obs(next_obs, name)
            self.agents[name].buffer.store(
                lo, np.array([action[idx]]),
                float(local_rewards[idx]), lo_next, float(done)
            )

        # Supervisor — FIX-B: scale reward to env range
        sup_obs      = np.concatenate([obs,      prev_local_rewards])
        sup_next_obs = np.concatenate([next_obs, local_rewards])
        sup_reward   = float(np.dot(omega, local_rewards)) * ENV_REWARD_SCALE
        self.supervisor.buffer.store(
            sup_obs, omega, sup_reward, sup_next_obs, float(done)
        )

    # ------------------------------------------------------------------
    def update_all(self, batch_size: int = 256) -> dict:
        losses = {}

        for name, agent in self.agents.items():
            info = agent.update(batch_size)
            if info:
                losses[name] = info

        # FIX-A: supervisor trains only after its own dedicated warm-up
        if self.supervisor.buffer.size >= SUP_WARMUP_SIZE:
            sup_info = self.supervisor.update(batch_size)
            if sup_info:
                losses["supervisor"] = sup_info

        return losses

    # ------------------------------------------------------------------
    def compute_local_rewards(self, info: dict) -> np.ndarray:
        """
        FIX-C: non-overlapping reward decomposition.
          BESS  → discharge value + degradation
          EV    → V2G discharge value only
          Load  → comfort deviation penalty only
          Grid  → ALL residual import cost + load-loss penalty (sole owner)
        """
        lam    = info.get("tariff",    0.1)
        p_bess = info.get("p_bess",    0.0)
        p_ev   = info.get("p_ev",      0.0)
        p_flex = info.get("p_flex",    info.get("p_load", 30.0))
        p_grid = info.get("p_grid",    0.0)
        ll     = info.get("load_loss", 0.0)

        from microgrid_env import MicrogridEnv as _E

        r_bess = (lam * p_bess * _E.DT) \
                 - _E.GAMMA * (abs(p_bess) / _E.P_BESS_MAX) ** _E.KAPPA
        r_ev   = (lam * max(0.0, p_ev) * _E.DT) * 0.5
        r_load = -abs(p_flex - 30.0) * _E.ZETA
        r_grid = -(lam * max(0.0, p_grid) * _E.DT) - ll * 1.0

        rewards = np.array([r_bess, r_ev, r_load, r_grid], dtype=np.float32)
        scale   = max(float(np.abs(rewards).max()), 1.0)
        return np.clip(rewards / scale, -1.0, 1.0)

    # ------------------------------------------------------------------
    def save_if_best(self, eval_reward: float, save_dir, method: str = "hma") -> bool:
        """FIX-E: save weights only when eval_reward beats running best."""
        if eval_reward > self.best_eval_reward:
            self.best_eval_reward = eval_reward
            _save_hma_weights(self, save_dir, method)
            return True
        return False


# ---------------------------------------------------------------------------
# Flat MA-DRL baseline — unchanged
# ---------------------------------------------------------------------------
class FlatMADRL:
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
            lo = local_obs(obs, name)
            if isinstance(self.agents[name], TD3Agent):
                action[idx] = self.agents[name].select_action(lo, 0.1 if explore else 0.0)[0]
            else:
                action[idx] = self.agents[name].select_action(lo, deterministic=not explore)[0]
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
# Single-Agent DRL baseline — unchanged
# ---------------------------------------------------------------------------
class SingleAgentDRL:
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
    e = np.exp(x)
    return e / e.sum()


def _save_hma_weights(controller: HMADRLFramework, save_dir, method: str):
    import torch
    w = {}
    for name, agent in controller.agents.items():
        w[f"{name}_actor"]  = agent.actor.state_dict()
        w[f"{name}_critic"] = agent.critic.state_dict()
    w["supervisor_actor"]  = controller.supervisor.actor.state_dict()
    w["supervisor_critic"] = controller.supervisor.critic.state_dict()
    torch.save(w, save_dir / f"{method}_weights.pt")
    print(f"  Weights saved → {save_dir}/{method}_weights.pt")
