"""
adaptive_reward.py  — v2 (fixed)

FIXES vs v1:
  FIX-1  Exploration std changed from proportional (weight_means*0.05)
         to fixed (0.02).  Proportional std caused log_prob to grow as
         weights grew, eventually producing gradients that blew out the
         network, locking ARW_loss at -100 permanently.
  FIX-2  Raw returns clipped to [-50, 50] before normalisation.
  FIX-3  Std floor raised to 1.0 (was 1e-8) so normalisation never
         divides by near-zero.
  FIX-4  Advantage clip tightened from ±5 to ±2.
  FIX-5  Policy and entropy loss clipped separately before summing
         (tighter bounds: ±10 and ±1 respectively).
  FIX-6  Gradient clip tightened from 0.5 to 0.1.
  FIX-7  Network weights clamped to [-5, 5] after every update to
         prevent runaway parameters.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


BASE_WEIGHTS      = np.array([1.0, 0.3, 0.2, 0.5], dtype=np.float32)
BASE_WEIGHTS_NORM = BASE_WEIGHTS / BASE_WEIGHTS.sum()
WEIGHT_SCALE      = float(BASE_WEIGHTS.sum())   # 2.0


class AdaptiveRewardWeightNetwork(nn.Module):
    def __init__(self, obs_dim: int = 7, hidden: Tuple[int, int] = (32, 16)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], 4),
        )
        # Initialise so the network starts at the paper's fixed weights
        with torch.no_grad():
            self.net[-1].bias.copy_(
                torch.tensor(np.log(BASE_WEIGHTS_NORM + 1e-8).astype(np.float32))
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits  = self.net(obs)
        weights = F.softmax(logits, dim=-1) * WEIGHT_SCALE
        return weights


class AdaptiveRewardWeighter:
    """
    REINFORCE-trained wrapper around AdaptiveRewardWeightNetwork.

    Usage:
        arw = AdaptiveRewardWeighter()

        # each timestep:
        weights = arw.get_weights(obs)          # → np.ndarray [w_c,w_b,w_e,w_s]
        arw.record(reward)

        # each episode end:
        info = arw.update()
    """

    def __init__(
        self,
        obs_dim:         int   = 7,
        lr:              float = 1e-3,
        gamma:           float = 0.99,
        entropy_coef:    float = 0.01,
        device:          str   = "cpu",
        warmup_episodes: int   = 50,
    ):
        self.device          = torch.device(device)
        self.gamma           = gamma
        self.entropy_coef    = entropy_coef
        self.warmup_episodes = warmup_episodes
        self._episode        = 0

        self.net = AdaptiveRewardWeightNetwork(obs_dim).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

        self._log_probs: list = []
        self._rewards:   list = []
        self._entropies: list = []

        self._baseline       = 0.0
        self._baseline_alpha = 0.05

    @property
    def is_warming_up(self) -> bool:
        return self._episode < self.warmup_episodes

    # ------------------------------------------------------------------
    def get_weights(self, obs: np.ndarray) -> np.ndarray:
        if self.is_warming_up:
            return BASE_WEIGHTS.copy()

        obs_t        = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        weight_means = self.net(obs_t).squeeze(0)

        # FIX-1: fixed std = 0.02 (was weight_means * 0.05, which grew without bound)
        std   = torch.full_like(weight_means, 0.02)
        noise = torch.randn_like(weight_means) * std

        weights_sampled = (weight_means + noise).clamp(min=0.05)
        weights_sampled = weights_sampled / weights_sampled.sum() * WEIGHT_SCALE

        dist     = torch.distributions.Normal(weight_means, std)
        log_prob = dist.log_prob(weights_sampled.detach()).sum()
        entropy  = dist.entropy().sum()

        self._log_probs.append(log_prob)
        self._entropies.append(entropy)

        return weights_sampled.detach().cpu().numpy()

    # ------------------------------------------------------------------
    def record(self, reward: float):
        self._rewards.append(float(reward))

    # ------------------------------------------------------------------
    def update(self) -> dict:
        self._episode += 1

        if self.is_warming_up or len(self._log_probs) == 0:
            self._rewards.clear()
            self._log_probs.clear()
            self._entropies.clear()
            return {"arw_loss": 0.0, "arw_episode": self._episode}

        T       = len(self._rewards)
        returns = np.zeros(T, dtype=np.float32)
        G       = 0.0
        for t in reversed(range(T)):
            G          = self._rewards[t] + self.gamma * G
            returns[t] = G

        # FIX-2: clip raw returns before normalising
        returns  = np.clip(returns, -50.0, 50.0)
        ret_mean = returns.mean()
        # FIX-3: floor std at 1.0 to prevent division-by-near-zero
        ret_std  = max(float(returns.std()), 1.0)
        returns  = (returns - ret_mean) / ret_std

        self._baseline = (
            (1 - self._baseline_alpha) * self._baseline
            + self._baseline_alpha * ret_mean
        )

        ret_tensor   = torch.FloatTensor(returns).to(self.device)
        n            = min(len(self._log_probs), T)
        policy_loss  = torch.zeros(1, device=self.device)
        entropy_loss = torch.zeros(1, device=self.device)

        for i in range(n):
            # FIX-4: tighter advantage clip (was ±5)
            advantage    = max(-2.0, min(2.0, float(ret_tensor[i]) - self._baseline))
            policy_loss  = policy_loss  - self._log_probs[i] * advantage
            entropy_loss = entropy_loss - self._entropies[i]

        # FIX-5: clip components separately before summing (much tighter bounds)
        pl         = torch.clamp(policy_loss  / n, -10.0, 10.0)
        el         = torch.clamp(self.entropy_coef * entropy_loss / n, -1.0, 1.0)
        total_loss = pl + el

        self.opt.zero_grad()
        total_loss.backward()
        # FIX-6: tighter gradient clip (was 0.5)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.1)
        self.opt.step()

        # FIX-7: clamp network weights to prevent runaway parameters
        with torch.no_grad():
            for p in self.net.parameters():
                p.clamp_(-5.0, 5.0)

        loss_val = total_loss.item()
        self._rewards.clear()
        self._log_probs.clear()
        self._entropies.clear()

        return {
            "arw_loss":     loss_val,
            "arw_episode":  self._episode,
            "arw_return":   float(ret_mean),
            "arw_baseline": float(self._baseline),
        }

    # ------------------------------------------------------------------
    def state_dict(self):
        return {
            "net":      self.net.state_dict(),
            "opt":      self.opt.state_dict(),
            "episode":  self._episode,
            "baseline": self._baseline,
        }

    def load_state_dict(self, d: dict):
        self.net.load_state_dict(d["net"])
        self.opt.load_state_dict(d["opt"])
        self._episode  = d.get("episode",  0)
        self._baseline = d.get("baseline", 0.0)
