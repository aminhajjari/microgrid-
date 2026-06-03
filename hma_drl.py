"""
hma_drl.py  (v5 — ARW integrated)
==========
Hierarchical Multi-Agent DRL (HMA-DRL) framework.

==========================================================================
CHANGES IN v5  (on top of v4)
==========================================================================

ARW  Adaptive Reward Weighting — novelty contribution.

     HMADRLFramework now owns an AdaptiveRewardWeighter (ARWN) instance.
     Each timestep, get_reward_weights(obs) is called and the 4-element
     weight array is passed to env.step(action, reward_weights=...).

     The ARWN has a 50-episode warm-up during which it outputs the paper's
     original fixed weights exactly — so the smoke test (60 episodes) still
     passes with the same baseline behaviour.  After warm-up the ARWN
     starts learning context-aware weights.

     CRITICAL safeguard: weights are clipped to [0.01, 10.0] inside
     microgrid_env.py so a bad early ARWN output cannot produce the
     extreme negative rewards (-32) seen in the previous run.

     SA and Flat baselines pass reward_weights=None to env.step(), so
     they are completely unaffected — fair comparison preserved.
"""

# VERSION: hma_drl v5

from __future__ import annotations

import numpy as np
import torch

from agents import TD3Agent, SACAgent
from adaptive_reward import AdaptiveRewardWeighter
from microgrid_constants import DT, GAMMA, P_BESS_MAX, KAPPA, ZETA

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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUP_WARMUP_SIZE        = 2000
ENV_REWARD_SCALE       = 10.0
OMEGA_BIAS_SCALE       = 0.15
OMEGA_MODULATED_AGENTS = ["bess", "ev"]
LOCAL_REWARD_SCALE     = 4.0


class HMADRLFramework:
    """Hierarchical multi-agent DRL controller with Adaptive Reward Weighting."""

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

        # ARW: Adaptive Reward Weighter (novelty)
        # warmup_episodes=50 means first 50 episodes use fixed paper weights
        # exactly — smoke test (60 ep) will be almost entirely in warm-up
        self.arw = AdaptiveRewardWeighter(
            obs_dim          = OBS_DIM,
            lr               = 1e-3,
            gamma            = 0.99,
            entropy_coef     = 0.01,
            device           = d,
            warmup_episodes  = 50,
        )

        self._last_omega      = _softmax(np.zeros(N_AGENTS))
        self._device          = torch.device(d)
        self.best_eval_reward = -np.inf

    # ------------------------------------------------------------------
    def get_reward_weights(self, obs: np.ndarray) -> np.ndarray:
        """
        Return adaptive reward weights [w_c, w_b, w_e, w_s].
        Called once per timestep before env.step().
        During ARW warm-up returns paper's original fixed weights.
        """
        return self.arw.get_weights(obs)

    # ------------------------------------------------------------------
    def record_reward(self, reward: float):
        """Pass the step reward to ARW for REINFORCE training."""
        self.arw.record(reward)

    # ------------------------------------------------------------------
    def update_arw(self) -> dict:
        """Call once per episode end to update ARW weights."""
        return self.arw.update()

    # ------------------------------------------------------------------
    def select_actions(
        self,
        obs: np.ndarray,
        local_rewards: np.ndarray | None = None,
        explore: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        for name, idx in LOCAL_ACT_IDX.items():
            lo      = local_obs(obs,      name)
            lo_next = local_obs(next_obs, name)
            scaled_reward = float(local_rewards[idx]) * LOCAL_REWARD_SCALE
            self.agents[name].buffer.store(
                lo, np.array([action[idx]]),
                scaled_reward, lo_next, float(done)
            )

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
        p_flex = info.get("p_flex",    info.get("p_load", 30.0))
        p_grid = info.get("p_grid",    0.0)
        ll     = info.get("load_loss", 0.0)

        

        r_bess = (lam * p_bess * DT) - GAMMA * (abs(p_bess) / P_BESS_MAX) ** KAPPA
        r_ev   = (lam * max(0.0, p_ev) * DT) * 0.5
        r_load = -abs(p_flex - 30.0) * ZETA
        r_grid = -(lam * max(0.0, p_grid) * DT) - ll * 1.0

        rewards = np.array([r_bess, r_ev, r_load, r_grid], dtype=np.float32)
        scale   = max(float(np.abs(rewards).max()), 1.0)
        return np.clip(rewards / scale, -1.0, 1.0)

    # ------------------------------------------------------------------
    def save_if_best(self, eval_reward: float, save_dir, method: str = "hma") -> bool:
        if eval_reward > self.best_eval_reward:
            self.best_eval_reward = eval_reward
            _save_hma_weights(self, save_dir, method)
            return True
        return False


# ---------------------------------------------------------------------------
# Flat MA-DRL baseline — unchanged, no ARW
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
# Single-Agent DRL baseline — unchanged, no ARW
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
    w["arw"]               = controller.arw.state_dict()   # save ARW weights too
    torch.save(w, save_dir / f"{method}_weights.pt")
    print(f"  Weights saved → {save_dir}/{method}_weights.pt")
