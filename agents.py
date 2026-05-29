"""
agents.py
=========
Lightweight TD3 and SAC implementations in pure PyTorch.
Now includes Prioritized Experience Replay (PER) as mentioned in the paper (Sec 2.5).
"""

import copy
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SumTree for Prioritized Replay
# ============================================================

class SumTree:
    """Binary sum tree for O(log n) sampling and priority updates."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0
        self.ptr = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def update(self, idx: int, priority: float):
        """Set priority of a leaf (index in data array)."""
        tree_idx = idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def add(self, priority: float, data):
        """Store new transition and its priority."""
        self.data[self.ptr] = data
        self.update(self.ptr, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self, s: float):
        """Return (data_idx, tree_idx, priority) for sampling."""
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1
        return data_idx, tree_idx, self.tree[tree_idx]


# ============================================================
# Prioritized Replay Buffer
# ============================================================

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) with annealed beta.
    Hyperparameters (can be adjusted):
        alpha = 0.6   (how much prioritization)
        beta_start = 0.4 (annealed to 1.0 over training)
        epsilon = 1e-6 (small constant to avoid zero priority)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        capacity: int = 1_000_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta = beta_start   # will be annealed externally

        self.tree = SumTree(capacity)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._frames = 0   # for beta annealing

        # Pre-allocate arrays for faster store (optional)
        self.obs_arr  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_arr  = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew_arr  = np.zeros((capacity, 1),       dtype=np.float32)
        self.nobs_arr = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done_arr = np.zeros((capacity, 1),       dtype=np.float32)

    def update_beta(self, frame: int):
        """Anneals beta from beta_start to 1.0 over beta_frames."""
        self._frames = frame
        self.beta = min(1.0, self.beta_start + frame * (1.0 - self.beta_start) / self.beta_frames)

    def store(self, obs, act, rew, nobs, done):
        """Store transition with maximal priority (to ensure it gets sampled)."""
        data = (obs.copy(), act.copy(), rew, nobs.copy(), done)
        idx = self.tree.ptr
        # Store in arrays
        self.obs_arr[idx]  = obs
        self.act_arr[idx]  = act
        self.rew_arr[idx]  = rew
        self.nobs_arr[idx] = nobs
        self.done_arr[idx] = done
        # Max priority = current max in tree or 1.0 if tree empty
        max_priority = self.tree.tree.max() if self.tree.size > 0 else 1.0
        self.tree.add(max_priority ** self.alpha, idx)

    def sample(self, batch_size: int, device: torch.device):
        """Return (obs, act, rew, nobs, done, weights, indices)."""
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, tree_idx, p = self.tree.get(s)
            indices.append(idx)
            priorities.append(p)
            # Retrieve stored data
            obs = self.obs_arr[idx]
            act = self.act_arr[idx]
            rew = self.rew_arr[idx]
            nobs = self.nobs_arr[idx]
            done = self.done_arr[idx]
            batch.append((obs, act, rew, nobs, done))

        # Compute importance-sampling weights
        total = self.tree.total()
        if total == 0:
            probs = np.ones(len(priorities)) / len(priorities)
        else:
            probs = np.array(priorities) / total
        weights = (self.tree.size * probs) ** (-self.beta)
        weights /= weights.max()   # normalize for stability
        weights = torch.FloatTensor(weights).to(device).unsqueeze(1)

        # Convert batch to tensors
        obs_b  = torch.FloatTensor(np.stack([b[0] for b in batch])).to(device)
        act_b  = torch.FloatTensor(np.stack([b[1] for b in batch])).to(device)
        rew_b  = torch.FloatTensor(np.stack([b[2] for b in batch])).to(device)
        nobs_b = torch.FloatTensor(np.stack([b[3] for b in batch])).to(device)
        done_b = torch.FloatTensor(np.stack([b[4] for b in batch])).to(device)

        return obs_b, act_b, rew_b, nobs_b, done_b, weights, indices

    def update_priorities(self, indices: list, td_errors: np.ndarray):
        """Update priorities based on TD error (abs)."""
        for idx, td_err in zip(indices, td_errors):
            priority = (abs(td_err) + 1e-6) ** self.alpha
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.size
    
    def __len__(self):
        return self.tree.size

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


# ============================================================
# TD3 (updated with PER)
# ============================================================

class TD3Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(128, 64)):
        super().__init__()
        self.net = _mlp(obs_dim, act_dim, hidden)

    def forward(self, obs):
        return torch.tanh(self.net(obs))


class TD3Critic(nn.Module):
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
    Twin Delayed Deep Deterministic Policy Gradient with PER.
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
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100_000,
    ):
        self.gamma        = gamma
        self.tau          = tau
        self.policy_noise = policy_noise
        self.noise_clip   = noise_clip
        self.policy_delay = policy_delay
        self.device       = torch.device(device)
        self._updates     = 0
        self._frame       = 0   # for beta annealing

        self.actor         = TD3Actor(obs_dim, act_dim, hidden).to(self.device)
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic        = TD3Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Prioritized Replay Buffer
        self.buffer = PrioritizedReplayBuffer(
            obs_dim, act_dim, capacity=1_000_000,
            alpha=per_alpha, beta_start=per_beta_start,
            beta_frames=per_beta_frames
        )

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

        # Anneal beta for PER
        self.buffer.update_beta(self._frame)
        self._frame += 1

        # Sample with importance weights
        obs, act, rew, nobs, done, weights, indices = self.buffer.sample(batch_size, self.device)
        self._updates += 1

        with torch.no_grad():
            noise = (torch.randn_like(act) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_act = (self.actor_target(nobs) + noise).clamp(-1, 1)
            q1_t, q2_t = self.critic_target(nobs, next_act)
            q_target = rew + self.gamma * (1 - done) * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(obs, act)
        td_error1 = (q1 - q_target).detach().cpu().numpy().flatten()
        td_error2 = (q2 - q_target).detach().cpu().numpy().flatten()
        td_error = (np.abs(td_error1) + np.abs(td_error2)) / 2.0

        critic_loss = (weights * (F.mse_loss(q1, q_target, reduction='none') +
                                  F.mse_loss(q2, q_target, reduction='none'))).mean()

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
            _soft_update(self.actor,  self.actor_target,  self.tau)
            _soft_update(self.critic, self.critic_target, self.tau)

        # Update priorities in buffer
        self.buffer.update_priorities(indices, td_error)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_val}


# ============================================================
# SAC (updated with PER)
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
    Soft Actor-Critic with automatic entropy tuning and PER.
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
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100_000,
    ):
        self.gamma  = gamma
        self.tau    = tau
        self.device = torch.device(device)
        self._frame = 0

        self.actor  = SACGaussianActor(obs_dim, act_dim, hidden).to(self.device)
        self.critic = SACCritic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.target_entropy = target_entropy if target_entropy else -float(act_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        # Prioritized Replay Buffer
        self.buffer = PrioritizedReplayBuffer(
            obs_dim, act_dim, capacity=1_000_000,
            alpha=per_alpha, beta_start=per_beta_start,
            beta_frames=per_beta_frames
        )

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

        # Anneal beta for PER
        self.buffer.update_beta(self._frame)
        self._frame += 1

        obs, act, rew, nobs, done, weights, indices = self.buffer.sample(batch_size, self.device)

        with torch.no_grad():
            nact, log_pi = self.actor.sample(nobs)
            q1_t, q2_t  = self.critic_target(nobs, nact)
            q_target = rew + self.gamma * (1 - done) * (
                torch.min(q1_t, q2_t) - self.alpha.detach() * log_pi
            )

        q1, q2 = self.critic(obs, act)
        td_error1 = (q1 - q_target).detach().cpu().numpy().flatten()
        td_error2 = (q2 - q_target).detach().cpu().numpy().flatten()
        td_error = (np.abs(td_error1) + np.abs(td_error2)) / 2.0

        critic_loss = (weights * (F.mse_loss(q1, q_target, reduction='none') +
                                  F.mse_loss(q2, q_target, reduction='none'))).mean()

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

        # Update priorities
        self.buffer.update_priorities(indices, td_error)

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
