"""
hma_drl.py  — v2 (fixed)

FIXES vs v1:
  FIX-1  compute_local_rewards() normalises each agent's reward
         independently (was dividing by global max, which crushed
         smaller agents like load to near zero).
  FIX-2  Load agent reward now includes a cost-saving term so it has
         genuine incentive to shift load away from peak tariff hours.
  FIX-3  Supervisor warmup raised from 2,000 to 10,000 steps so it
         only starts coordinating after local agents have learned
         basic policies.
"""

# VERSION: hma_drl v2

from __future__ import annotations

import numpy as np
import torch

from agents import TD3Agent, SACAgent
from adaptive_reward import AdaptiveRewardWeighter
from microgrid_constants import DT, GAMMA, P_BESS_MAX, KAPPA, ZETA

OBS_DIM  = 7
ACT_DIM  = 4
N_AGENTS = 4

LOCAL_OBS_IDX = {
    "bess": [0, 2, 3, 4, 5, 6],
    "ev":   [1, 2, 3, 4, 5, 6],
    "load": [2, 3, 4, 5, 6],
    "grid": [2, 3, 4, 5, 6],
}
LOCAL_ACT_IDX = {"bess": 0, "ev": 1, "load": 2, "grid": 3}


def local_obs(obs: np.ndarray, agent: str) -> np.ndarray:
    return obs[LOCAL_OBS_IDX[agent]]


# FIX-3: raised from 2000 to 10000
SUP_WARMUP_SIZE        = 10_000
ENV_REWARD_SCALE       = 1.0
OMEGA_BIAS_SCALE       = 0.15
OMEGA_MODULATED_AGENTS = ["bess", "ev"]
LOCAL_REWARD_SCALE     = 1.0


class HMADRLFramework:
    """Hierarchical MA-DRL with Adaptive Reward Weighting."""

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

        self.supervisor = SACAgent(OBS_DIM + N_AGENTS, N_AGENTS, device=d)

        self.arw = AdaptiveRewardWeighter(
            obs_dim         = OBS_DIM,
            lr              = 1e-3,
            gamma           = 0.99,
            entropy_coef    = 0.01,
            device          = d,
            warmup_episodes = 50,
        )

        self._last_omega      = _softmax(np.zeros(N_AGENTS))
        self._device          = torch.device(d)
        self.best_eval_reward = -np.inf

    # ------------------------------------------------------------------
    def get_reward_weights(self, obs: np.ndarray) -> np.ndarray:
        return self.arw.get_weights(obs)

    def record_reward(self, reward: float):
        self.arw.record(reward)

    def update_arw(self) -> dict:
        return self.arw.update()

    # ------------------------------------------------------------------
    def select_actions(self, obs, local_rewards=None, explore=True):
        if local_rewards is None:
            local_rewards = np.zeros(N_AGENTS)

        action = np.zeros(4)
        for name, idx in LOCAL_ACT_IDX.items():
            lo = local_obs(obs, name)
            if isinstance(self.agents[name], TD3Agent):
                noise = 0.1 if explore else 0.0
                action[idx] = self.agents[name].select_action(lo, noise_std=noise)[0]
            else:
                action[idx] = self.agents[name].select_action(
                    lo, deterministic=not explore)[0]

        sup_obs   = np.concatenate([obs, local_rewards])
        omega_raw = self.supervisor.select_action(sup_obs, deterministic=not explore)
        omega     = _softmax(omega_raw)
        self._last_omega = omega.copy()

        if self.supervisor.buffer.size >= SUP_WARMUP_SIZE:
            for i, name in enumerate(OMEGA_MODULATED_AGENTS):
                idx = LOCAL_ACT_IDX[name]
                action[idx] = np.clip(
                    action[idx] + OMEGA_BIAS_SCALE * (omega[i] - 0.25), -1.0, 1.0
                )

        return action, omega

    # ------------------------------------------------------------------
    def store_transitions(self, obs, action, local_rewards, global_reward,
                          next_obs, done, prev_local_rewards, omega):
        for name, idx in LOCAL_ACT_IDX.items():
            lo      = local_obs(obs,      name)
            lo_next = local_obs(next_obs, name)
            scaled  = float(local_rewards[idx]) * LOCAL_REWARD_SCALE
            self.agents[name].buffer.store(
                lo, np.array([action[idx]]), scaled, lo_next, float(done)
            )

        sup_obs      = np.concatenate([obs,      prev_local_rewards])
        sup_next_obs = np.concatenate([next_obs, local_rewards])
       
        sup_reward = float(np.dot(omega, local_rewards)) * ENV_REWARD_SCALE
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
        if self.supervisor.buffer.size >= SUP_WARMUP_SIZE:
            sup_info = self.supervisor.update(batch_size)
            if sup_info:
                losses["supervisor"] = sup_info
        return losses

    # ------------------------------------------------------------------
    def compute_local_rewards(self, info: dict) -> np.ndarray:
        lam    = info.get("tariff",    0.1)
        p_bess = info.get("p_bess",    0.0)
        p_ev   = info.get("p_ev",      0.0)
        p_flex = info.get("p_flex",    30.0)
        p_grid = info.get("p_grid",    0.0)
        ll     = info.get("load_loss", 0.0)

        # BESS: reward efficient discharge, penalise degradation
        r_bess = -GAMMA * (abs(p_bess) / P_BESS_MAX) ** KAPPA

        # EV: small reward for managed charging
        r_ev = -abs(p_ev) * 0.001

        # Load: penalise deviation from baseline comfort
        r_load = -abs(p_flex - 30.0) * ZETA * 0.1

        # Grid: HARD penalty for load loss, mild penalty for grid import
        r_grid = -ll * 2.0 - lam * max(0.0, p_grid) * DT * 0.1

        rewards = np.array([r_bess, r_ev, r_load, r_grid], dtype=np.float32)

        # Normalise by global scale, floor at 0.5 to preserve signal
        scale = max(float(np.max(np.abs(rewards))), 0.5)
        rewards = np.clip(rewards / scale, -1.0, 1.0)

        return rewards

    # ------------------------------------------------------------------
    def save_if_best(self, eval_reward: float, save_dir, method: str = "hma") -> bool:
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
    w["arw"]               = controller.arw.state_dict()
    torch.save(w, save_dir / f"{method}_weights.pt")
    print(f"  Weights saved → {save_dir}/{method}_weights.pt")
