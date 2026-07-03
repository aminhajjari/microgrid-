"""
adaptive_reward.py  — v4 (Dirichlet REINFORCE — correct score function)

NEW in v4:
  CHANGE-D  The sampler is now a Dirichlet over the weight simplex instead of
            a Gaussian on a clamped+renormalised vector. The old code computed
            Normal(means, 0.02).log_prob(renormalised_sample) — the log-prob of
            a sample that was NOT a draw from that Normal, so the score-function
            estimator was biased and its magnitude exploded as the weights
            saturated (this is what pinned the reported ARW loss at the +10
            clamp ceiling). The network now outputs Dirichlet concentrations;
            log_prob and entropy are exact, simplex-constrained, and the
            entropy term is finally a live function of the parameters (the old
            fixed-std Gaussian had constant entropy → zero exploration gradient).
  CHANGE-E  Grad-clip relaxed 0.1 → 1.0 and the policy-loss safety clamp widened
            ±10 → ±50 so it is a guard rail, not an active constraint.

PREVIOUS (v3):
  CHANGE-1  AdaptiveRewardWeightNetwork now accepts a scenario one-hot
            embedding concatenated to the obs input (obs_dim + N_SCENARIOS).
            This lets the ARW learn different weight priorities per scenario
            (normal / crit_load / pv_outage / dynamic_price / high_res).

  CHANGE-2  AdaptiveRewardWeighter.get_weights() gains a `scenario` kwarg
            (default "normal" — fully backward-compatible).
            The one-hot is built internally; callers just pass a string.

  CHANGE-3  AdaptiveRewardWeighter.__init__() gains an `n_scenarios` param
            and passes obs_dim + n_scenarios to the network.

All v2 fixes (FIX-1 … FIX-7) are preserved unchanged.
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

# ── CHANGE-D: Dirichlet concentration controls (exploration on the simplex) ─
CONC_SCALE = 5.0     # larger => sharper Dirichlet => less exploration
MIN_CONC   = 1.0     # floor on every concentration so alpha > 0 always
MAX_CONC   = 100.0   # numerical safety ceiling
# ──────────────────────────────────────────────────────────────────────────

# ── CHANGE-1 ──────────────────────────────────────────────────────────────
SCENARIOS   = ["normal", "crit_load", "pv_outage", "dynamic_price", "high_res"]
N_SCENARIOS = len(SCENARIOS)                          # 5
SCENARIO_IDX = {s: i for i, s in enumerate(SCENARIOS)}
# ──────────────────────────────────────────────────────────────────────────


class AdaptiveRewardWeightNetwork(nn.Module):
    # ── CHANGE-1: input is now obs_dim + n_scenarios ───────────────────────
    def __init__(self, obs_dim: int = 7,
                 n_scenarios: int = N_SCENARIOS,
                 hidden: Tuple[int, int] = (32, 16)):
        super().__init__()
        in_dim = obs_dim + n_scenarios                # e.g. 7 + 5 = 12
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
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
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # CHANGE-D: output positive Dirichlet concentrations, not weight means.
        logits = self.net(obs)
        alpha  = F.softplus(logits) * CONC_SCALE + MIN_CONC
        return alpha.clamp(max=MAX_CONC)


class AdaptiveRewardWeighter:
    """
    REINFORCE-trained wrapper around AdaptiveRewardWeightNetwork.
    Now scenario-aware: pass scenario="crit_load" etc. to get_weights().

    Usage:
        arw = AdaptiveRewardWeighter()
        arw.set_scenario("crit_load")       # ← set once per training run

        # each timestep:
        weights = arw.get_weights(obs)      # → np.ndarray [w_c,w_b,w_e,w_s]
        arw.record(reward)

        # each episode end:
        info = arw.update()
    """

    def __init__(
        self,
        obs_dim:         int   = 7,
        # ── CHANGE-3: new param, default keeps backward-compat ────────────
        n_scenarios:     int   = N_SCENARIOS,
        # ──────────────────────────────────────────────────────────────────
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
        # ── CHANGE-2: store current scenario ──────────────────────────────
        self._scenario       = "normal"
        self.n_scenarios     = n_scenarios
        # ──────────────────────────────────────────────────────────────────

        # ── CHANGE-3: pass n_scenarios to network ─────────────────────────
        self.net = AdaptiveRewardWeightNetwork(obs_dim, n_scenarios).to(self.device)
        # ──────────────────────────────────────────────────────────────────
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

        self._log_probs: list = []
        self._rewards:   list = []
        self._entropies: list = []

        self._baseline       = 0.0
        self._baseline_alpha = 0.05
        self._run_var        = 1.0     #  cross-episode return variance
        self._baseline_init  = False   #  seed baseline on first update5

    # ── CHANGE-2: new setter ───────────────────────────────────────────────
    def set_scenario(self, scenario: str):
        """Call once when the scenario changes (e.g. at the start of a run)."""
        if scenario not in SCENARIO_IDX:
            raise ValueError(f"Unknown scenario '{scenario}'. "
                             f"Valid: {list(SCENARIO_IDX.keys())}")
        self._scenario = scenario
    # ──────────────────────────────────────────────────────────────────────

    # ── CHANGE-2: internal helper ──────────────────────────────────────────
    def _scenario_onehot(self) -> np.ndarray:
        oh = np.zeros(self.n_scenarios, dtype=np.float32)
        oh[SCENARIO_IDX[self._scenario]] = 1.0
        return oh
    # ──────────────────────────────────────────────────────────────────────

    @property
    def is_warming_up(self) -> bool:
        return self._episode < self.warmup_episodes

    # ------------------------------------------------------------------
    def get_weights(self, obs: np.ndarray,
                    scenario: str = None) -> np.ndarray:
        # ── CHANGE-2: scenario kwarg overrides stored scenario if given ────
        if scenario is not None:
            self._scenario = scenario
        # ──────────────────────────────────────────────────────────────────

        if self.is_warming_up:
            return BASE_WEIGHTS.copy()

        # ── CHANGE-2: build augmented obs with scenario one-hot ───────────
        obs_aug = np.concatenate([obs, self._scenario_onehot()])
        obs_t   = torch.FloatTensor(obs_aug).unsqueeze(0).to(self.device)
        # ──────────────────────────────────────────────────────────────────

        # ── CHANGE-D: sample weights from a Dirichlet on the simplex ───────
        # network outputs concentrations; sample p ~ Dir(alpha), weights = p*SCALE.
        # log_prob / entropy are exact and the score function no longer explodes.
        alpha = self.net(obs_t).squeeze(0)               # (4,), all > 0
        dist  = torch.distributions.Dirichlet(alpha)
        p     = dist.sample()                            # simplex sample, no grad
        log_prob = dist.log_prob(p)                      # exact, grad wrt alpha
        entropy  = dist.entropy()                        # live function of alpha

        self._log_probs.append(log_prob)
        self._entropies.append(entropy)

        weights = (p * WEIGHT_SCALE)
        return weights.detach().cpu().numpy()
        # ──────────────────────────────────────────────────────────────────

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

        #  units-consistent, cross-episode whitening.
        # Old code standardized returns per-episode (mean 0, std 1) and then
        # subtracted a baseline EMA kept in RAW KPI units (≈ −45), so the
        # advantage was pinned at the +2 clip for every sample → REINFORCE
        # reinforced whatever was sampled → undirected concentration drift.
        returns  = np.clip(returns, -50.0, 50.0)
        ret_mean = float(returns.mean())
        if not self._baseline_init:                 # avoid the same transient
            self._baseline      = ret_mean
            self._baseline_init = True
        self._baseline = (
            (1 - self._baseline_alpha) * self._baseline
            + self._baseline_alpha * ret_mean
        )
        dev = returns - self._baseline
        self._run_var = (
            (1 - self._baseline_alpha) * self._run_var
            + self._baseline_alpha * float((dev ** 2).mean())
        )
        ret_std = max(float(np.sqrt(self._run_var)), 0.01)
        returns = dev / ret_std

        ret_tensor   = torch.FloatTensor(returns).to(self.device)
        n            = min(len(self._log_probs), T)
        policy_loss  = torch.zeros(1, device=self.device)
        entropy_loss = torch.zeros(1, device=self.device)

        for i in range(n):
            
            advantage    = max(-2.0, min(2.0, float(ret_tensor[i])))  # already baselined
            policy_loss  = policy_loss  - self._log_probs[i] * advantage
            entropy_loss = entropy_loss - self._entropies[i]

        # CHANGE-E: widened guard rail (was ±10, which pinned the old loss)
        pl         = torch.clamp(policy_loss  / n, -50.0, 50.0)
        el         = torch.clamp(self.entropy_coef * entropy_loss / n, -1.0, 1.0)
        total_loss = pl + el

        self.opt.zero_grad()
        total_loss.backward()
        # CHANGE-E: grad clip relaxed 0.1 → 1.0 (Dirichlet grads are bounded)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()

        # FIX-7
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
            # ── CHANGE-2: log current scenario for monitoring ──────────────
            "arw_scenario": self._scenario,
            # ──────────────────────────────────────────────────────────────
        }

    # ------------------------------------------------------------------
    def state_dict(self):
        return {
            "net":      self.net.state_dict(),
            "opt":      self.opt.state_dict(),
            "episode":  self._episode,
            "baseline": self._baseline,
            # ── CHANGE-2: persist scenario so checkpoints reload cleanly ───
            "scenario": self._scenario,
            # ──────────────────────────────────────────────────────────────
        }

    def load_state_dict(self, d: dict):
        self.net.load_state_dict(d["net"])
        self.opt.load_state_dict(d["opt"])
        self._episode  = d.get("episode",  0)
        self._baseline = d.get("baseline", 0.0)
        # ── CHANGE-2: restore scenario from checkpoint ─────────────────────
        self._scenario = d.get("scenario", "normal")
        # ──────────────────────────────────────────────────────────────────
