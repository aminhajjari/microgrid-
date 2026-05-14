"""
agents.py
=========
Lightweight TD3 and SAC implementations in pure PyTorch.
No external RL library required — mirrors the paper's algorithm choices:
  - TD3  → BESS agent, EV agent        (policy smoothness / overestimation)
  - SAC  → Load agent, Grid agent      (entropy exploration)
"""

import copy
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Shared utilities
# ============================================================

def _mlp(in_dim: int, out_dim: int, hidden: Tuple[int, ...] = (128, 64)) -> nn.Sequential:
    layers: list = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ReplayBuffer:
    """Uniform experience replay (supports prioritised extension)."""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 1_000_000):
        self.cap = capacity
        self.ptr = 0
        self.size = 0
        self.obs  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.nobs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act  = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew  = np.zeros((capacity, 1),       dtype=np.float32)
        self.done = np.zeros((capacity, 1),       dtype=np.float32)

    def store(self, obs, act, rew, nobs, done):
        self.obs[self.ptr]  = obs
        self.act[self.ptr]  = act
        self.rew[self.ptr]  = rew
        self.nobs[self.ptr] = nobs
        self.done[self.ptr] = done
        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch)
        return (
            torch.FloatTensor(self.obs[idx]).to(device),
            torch.FloatTensor(self.act[idx]).to(device),
            torch.FloatTensor(self.rew[idx]).to(device),
            torch.FloatTensor(self.nobs[idx]).to(device),
            torch.FloatTensor(self.done[idx]).to(device),
        )

    def __len__(self):
        return self.size


# ============================================================
# TD3
# ============================================================

class TD3Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(128, 64)):
        super().__init__()
        self.net = _mlp(obs_dim, act_dim, hidden)

    def forward(self, obs):
        return torch.tanh(self.net(obs))


class TD3Critic(nn.Module):
    """Twin Q-networks."""
    def __init__(self, obs_dim, act_dim, hidden=(128, 64)):
        super().__init__()
        self.q1 = _mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = _mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_val(self, obs, act):
        return self.q1(torch.cat([obs, act], dim=-1))


class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient.
    Hyperparameters match paper: lr=3e-4, γ=0.99, τ=0.005,
    policy noise σ=0.2, noise clip c=0.5, policy delay=2.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Tuple[int, ...] = (128, 64),
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        device: str = "cpu",
    ):
        self.gamma        = gamma
        self.tau          = tau
        self.policy_noise = policy_noise
        self.noise_clip   = noise_clip
        self.policy_delay = policy_delay
        self.device       = torch.device(device)
        self._updates     = 0

        self.actor         = TD3Actor(obs_dim, act_dim, hidden).to(self.device)
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic        = TD3Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = ReplayBuffer(obs_dim, act_dim)

    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            act   = self.actor(obs_t).cpu().numpy()[0]
        if noise_std > 0:
            act += noise_std * np.random.randn(*act.shape)
        return act.clip(-1, 1)

    # ------------------------------------------------------------------
    def update(self, batch_size: int = 256) -> dict:
        if len(self.buffer) < batch_size:
            return {}
        obs, act, rew, nobs, done = self.buffer.sample(batch_size, self.device)
        self._updates += 1

        with torch.no_grad():
            noise = (torch.randn_like(act) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_act = (self.actor_target(nobs) + noise).clamp(-1, 1)
            q1_t, q2_t = self.critic_target(nobs, next_act)
            q_target = rew + self.gamma * (1 - done) * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        actor_loss_val = 0.0
        if self._updates % self.policy_delay == 0:
            actor_loss = -self.critic.q1_val(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()
            actor_loss_val = actor_loss.item()
            # Soft update
            _soft_update(self.actor,  self.actor_target,  self.tau)
            _soft_update(self.critic, self.critic_target, self.tau)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_val}


# ============================================================
# SAC
# ============================================================

LOG_STD_MIN, LOG_STD_MAX = -5, 2


class SACGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(128, 64)):
        super().__init__()
        self.net     = _mlp(obs_dim, hidden[-1], hidden[:-1])
        self.mu_head = nn.Linear(hidden[-1], act_dim)
        self.ls_head = nn.Linear(hidden[-1], act_dim)

    def forward(self, obs):
        h      = F.relu(self.net(obs))
        mu     = self.mu_head(h)
        log_std = self.ls_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self(obs)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x_t  = dist.rsample()
        y_t  = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        return y_t, log_prob.sum(-1, keepdim=True)


class SACCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(128, 64)):
        super().__init__()
        self.q1 = _mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = _mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class SACAgent:
    """
    Soft Actor-Critic with automatic entropy tuning.
    Hyperparameters: lr=3e-4, γ=0.99, τ=0.005, target_entropy=-act_dim.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Tuple[int, ...] = (128, 64),
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: float = None,
        device: str = "cpu",
    ):
        self.gamma  = gamma
        self.tau    = tau
        self.device = torch.device(device)

        self.actor  = SACGaussianActor(obs_dim, act_dim, hidden).to(self.device)
        self.critic = SACCritic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning
        self.target_entropy = target_entropy if target_entropy else -float(act_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        self.buffer = ReplayBuffer(obs_dim, act_dim)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            if deterministic:
                mu, _ = self.actor(obs_t)
                return torch.tanh(mu).cpu().numpy()[0]
            act, _ = self.actor.sample(obs_t)
            return act.cpu().numpy()[0]

    # ------------------------------------------------------------------
    def update(self, batch_size: int = 256) -> dict:
        if len(self.buffer) < batch_size:
            return {}
        obs, act, rew, nobs, done = self.buffer.sample(batch_size, self.device)

        with torch.no_grad():
            nact, log_pi = self.actor.sample(nobs)
            q1_t, q2_t  = self.critic_target(nobs, nact)
            q_target = rew + self.gamma * (1 - done) * (
                torch.min(q1_t, q2_t) - self.alpha.detach() * log_pi
            )

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        new_act, log_pi = self.actor.sample(obs)
        q1_pi, q2_pi   = self.critic(obs, new_act)
        actor_loss = (self.alpha.detach() * log_pi - torch.min(q1_pi, q2_pi)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        _soft_update(self.critic, self.critic_target, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha.item(),
        }


# ============================================================
# Shared helper
# ============================================================

def _soft_update(src: nn.Module, tgt: nn.Module, tau: float):
    for sp, tp in zip(src.parameters(), tgt.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)