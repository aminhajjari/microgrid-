"""
adaptive_reward.py
==================
Adaptive Reward Weighting (ARW) module — Novelty contribution on top of HMA-DRL.

CONCEPT (for paper Section 2.4 extension):
    The base paper uses fixed weights [W_C=1.0, W_B=0.3, W_E=0.2, W_S=0.5] for the
    reward function (Eq. 9).  These constants cannot adapt to changing operating
    conditions — during a PV outage, grid cost matters more; during high cycling,
    battery health matters more.

    This module introduces an AdaptiveRewardWeightNetwork (ARWN) that:
      1. Observes the current environment state (7 dims)
      2. Outputs 4 reward weights [w_c, w_b, w_e, w_s] via softmax (sum = 1, scaled)
      3. Trains via REINFORCE (policy gradient): weights that led to higher global
         reward are reinforced, weights that led to poor reward are suppressed.

    The ARWN is integrated into HMADRLFramework.  It is called once per timestep
    and its output replaces the fixed W_C/W_B/W_E/W_S constants in microgrid_env.py.

INTEGRATION POINTS:
    - microgrid_env.py : step() receives optional weight dict, uses it instead of
                         class-level constants.
    - hma_drl.py       : HMADRLFramework holds an ARWN instance, calls it each step,
                         passes weights to env.step(), updates ARWN each episode end.

PAPER NOVELTY CLAIM:
    "Unlike prior works that rely on manually tuned, static reward weightings,
     the proposed ARW mechanism dynamically adjusts the multi-objective balance
     in real time based on observed system state, enabling context-aware
     optimization across diverse operating scenarios."
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Base weight initialisation — matches paper Eq. 9 defaults
# ---------------------------------------------------------------------------
BASE_WEIGHTS = np.array([1.0, 0.3, 0.2, 0.5], dtype=np.float32)   # W_C, W_B, W_E, W_S
BASE_WEIGHTS_NORM = BASE_WEIGHTS / BASE_WEIGHTS.sum()               # normalised to sum=1

# Scale factor so output weights have same magnitude as paper's originals
# (paper sums to 2.0; we normalise to 1.0 then multiply by this scale)
WEIGHT_SCALE = BASE_WEIGHTS.sum()   # = 2.0


# ---------------------------------------------------------------------------
# Neural Network
# ---------------------------------------------------------------------------

class AdaptiveRewardWeightNetwork(nn.Module):
    """
    Small 2-layer MLP: obs(7) -> hidden(32) -> hidden(16) -> weights(4).

    Output is passed through softmax then scaled so weights sum to WEIGHT_SCALE,
    preserving the same total reward magnitude as the base paper.

    Initialisation bias: softmax of zeros = uniform = [0.5, 0.5, 0.5, 0.5] * scale/4.
    We add a learned bias initialised to BASE_WEIGHTS_NORM * log(4) so the network
    starts close to the paper's original weights at episode 0.
    """

    def __init__(self, obs_dim: int = 7, hidden: Tuple[int, int] = (32, 16)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], 4),
        )
        # Initialise final layer bias so initial output ≈ paper's fixed weights
        with torch.no_grad():
            self.net[-1].bias.copy_(
                torch.tensor(
                    np.log(BASE_WEIGHTS_NORM + 1e-8).astype(np.float32)
                )
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return weight tensor of shape (..., 4), summing to WEIGHT_SCALE."""
        logits = self.net(obs)
        weights = F.softmax(logits, dim=-1) * WEIGHT_SCALE
        return weights


# ---------------------------------------------------------------------------
# REINFORCE trainer wrapper
# ---------------------------------------------------------------------------

class AdaptiveRewardWeighter:
    """
    Wraps ARWN with a REINFORCE update loop.

    Usage per episode:
        arw = AdaptiveRewardWeighter(device="cpu")

        # inside timestep loop:
        weights = arw.get_weights(obs)          # numpy array [w_c, w_b, w_e, w_s]
        # ... pass weights to env, get reward ...
        arw.record(reward)                       # store scalar reward for this step

        # at episode end:
        arw.update()                             # one REINFORCE gradient step
    """

    def __init__(
        self,
        obs_dim:      int   = 7,
        lr:           float = 1e-3,
        gamma:        float = 0.99,
        entropy_coef: float = 0.01,   # encourage weight diversity
        device:       str   = "cpu",
        warmup_episodes: int = 50,    # use fixed paper weights before training starts
    ):
        self.device          = torch.device(device)
        self.gamma           = gamma
        self.entropy_coef    = entropy_coef
        self.warmup_episodes = warmup_episodes
        self._episode        = 0

        self.net = AdaptiveRewardWeightNetwork(obs_dim).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

        # Per-episode storage
        self._log_probs: list[torch.Tensor] = []
        self._rewards:   list[float]        = []
        self._entropies: list[torch.Tensor] = []

        # Running baseline (exponential moving average of returns)
        self._baseline = 0.0
        self._baseline_alpha = 0.05

    # ------------------------------------------------------------------
    @property
    def is_warming_up(self) -> bool:
        return self._episode < self.warmup_episodes

    # ------------------------------------------------------------------
    def get_weights(self, obs: np.ndarray) -> np.ndarray:
        """
        Given current obs, return adaptive weights [w_c, w_b, w_e, w_s].
        During warm-up, returns paper's original fixed weights exactly.
        After warm-up, uses the ARWN with stochastic sampling for exploration.
        """
        if self.is_warming_up:
            return BASE_WEIGHTS.copy()

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        # Forward pass → mean weights (already scaled to WEIGHT_SCALE)
        weight_means = self.net(obs_t).squeeze(0)
        # Exploration noise (5% of magnitude, safe std)
        std = weight_means.detach() * 0.05 + 1e-6
        noise = torch.randn_like(weight_means) * std
        # Sample weights and clamp to avoid zero
        weights_sampled = (weight_means + noise).clamp(min=0.01)

        # 🔥 IMPORTANT: renormalize so sum stays constant
        weights_sampled = weights_sampled / weights_sampled.sum() * WEIGHT_SCALE

        # Build distribution for REINFORCE
        dist = torch.distributions.Normal(weight_means, std)

        log_prob = dist.log_prob(weights_sampled.detach()).sum()
        entropy  = dist.entropy().sum()

        # Store for update()
        self._log_probs.append(log_prob)
        self._entropies.append(entropy)

        return weights_sampled.detach().cpu().numpy()        

    # ------------------------------------------------------------------
    def record(self, reward: float):
        """Call once per timestep with the scalar environment reward."""
        self._rewards.append(float(reward))

    # ------------------------------------------------------------------
    def update(self) -> dict:
        """
        Call once per episode end.
        Performs one REINFORCE gradient step and resets episode buffers.
        Returns dict with loss info for logging.
        """
        self._episode += 1

        if self.is_warming_up or len(self._log_probs) == 0:
            self._rewards.clear()
            self._log_probs.clear()
            self._entropies.clear()
            return {"arw_loss": 0.0, "arw_episode": self._episode}

        # Compute discounted returns
        T = len(self._rewards)
        returns = np.zeros(T, dtype=np.float32)
        G = 0.0
        for t in reversed(range(T)):
            G = self._rewards[t] + self.gamma * G
            returns[t] = G

        # Normalise returns
        ret_mean = returns.mean()
        ret_std  = returns.std() + 1e-8
        returns  = (returns - ret_mean) / ret_std

        # Update running baseline
        self._baseline = (
            (1 - self._baseline_alpha) * self._baseline
            + self._baseline_alpha * ret_mean
        )

        # REINFORCE loss
        ret_tensor = torch.FloatTensor(returns).to(self.device)

        # Align log_probs with timesteps (may be fewer if some steps skipped)
        n = min(len(self._log_probs), T)
        policy_loss = torch.zeros(1, device=self.device)
        entropy_loss = torch.zeros(1, device=self.device)

        for i in range(n):
            advantage = ret_tensor[i] - self._baseline
            policy_loss  = policy_loss  - self._log_probs[i] * advantage
            entropy_loss = entropy_loss - self._entropies[i]   # maximise entropy

        total_loss = policy_loss / n + self.entropy_coef * entropy_loss / n

        self.opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.opt.step()

        loss_val = total_loss.item()

        # Reset buffers
        self._rewards.clear()
        self._log_probs.clear()
        self._entropies.clear()

        return {
            "arw_loss":    loss_val,
            "arw_episode": self._episode,
            "arw_return":  float(ret_mean),
            "arw_baseline": float(self._baseline),
        }

    # ------------------------------------------------------------------
    def state_dict(self):
        return {
            "net": self.net.state_dict(),
            "opt": self.opt.state_dict(),
            "episode": self._episode,
            "baseline": self._baseline,
        }

    def load_state_dict(self, d: dict):
        self.net.load_state_dict(d["net"])
        self.opt.load_state_dict(d["opt"])
        self._episode  = d.get("episode",  0)
        self._baseline = d.get("baseline", 0.0)
