"""
microgrid_env.py
================
OpenAI Gym-compatible microgrid environment for the HMA-DRL framework.
Models: PV generation, BESS, EV charging, flexible loads, grid interaction (TOU tariff).
All equations follow the paper numbering (Eq. 1–8).
"""


import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Helper: simple stochastic PV / load / EV profiles
# ---------------------------------------------------------------------------

def _solar_profile(T: int, seed: int = 0) -> np.ndarray:
    """Smooth bell-shaped PV curve with random noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    base = np.clip(np.sin(np.pi * (t - 6) / 12) * (t >= 6) * (t <= 18), 0, 1)
    return base * (1 + 0.15 * rng.standard_normal(T)).clip(0, 1)


def _load_profile(T: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = 0.4 + 0.3 * np.sin(2 * np.pi * np.arange(T) / T - np.pi / 2)
    return (base + 0.05 * rng.standard_normal(T)).clip(0.1, 1.0)


def _tou_tariff(T: int) -> np.ndarray:
    """Simple 3-band TOU: off-peak / mid / peak."""
    tariff = np.full(T, 0.08)          # off-peak  (0–7h)
    tariff[7:17] = 0.14                # mid-peak  (7–17h)
    tariff[17:21] = 0.22               # peak      (17–21h)
    return tariff


def _ev_availability(T: int, seed: int = 2) -> np.ndarray:
    """Binary plug-in status: mostly available 18–8h."""
    rng = np.random.default_rng(seed)
    alpha = np.zeros(T)
    for t in range(T):
        if t >= 18 or t <= 8:
            alpha[t] = 1 if rng.random() > 0.1 else 0
        else:
            alpha[t] = 1 if rng.random() > 0.8 else 0
    return alpha


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class MicrogridEnv(gym.Env):
    """
    Single-step (per time-slot) microgrid environment.

    State vector (7 dims):
        [soc_bess, soc_ev, pv_norm, load_norm, tariff_norm, hour_sin, hour_cos]

    Action vector (4 dims, all in [-1, 1]):
        [p_bess_norm, p_ev_norm, p_flex_norm, p_grid_norm]
        Positive  → discharge / import; Negative → charge / export / curtail.
    """

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------
    # Microgrid parameters (nominal, per-unit base = 100 kW)
    # ------------------------------------------------------------------
    E_BESS_MAX   = 200.0    # kWh
    P_BESS_MAX   = 50.0     # kW
    SOC_BESS_MIN = 0.1
    SOC_BESS_MAX = 0.9
    ETA_C_BESS   = 0.95
    ETA_D_BESS   = 0.95

    E_EV_MAX     = 60.0     # kWh
    P_EV_MAX     = 22.0     # kW
    SOC_EV_MIN   = 0.2
    SOC_EV_MAX   = 1.0
    ETA_C_EV     = 0.92
    ETA_D_EV     = 0.92

    P_GRID_MAX   = 80.0     # kW (import limit)
    P_GRID_MIN   = -50.0    # kW (export limit, negative)

    P_FLEX_MIN   = 0.0
    P_FLEX_MAX   = 30.0     # kW

    T = 24                  # 24-hour horizon, 1-hour slots
    DT = 1.0                # hours

    # Reward weights  (Eq. 9)
    W_C = 1.0
    W_B = 0.3
    W_E = 0.2
    W_S = 0.5

    # Degradation  (Eq. 11)
    GAMMA = 0.02            # USD / (kWh-cycle)
    KAPPA = 1.4

    # Penalty weights  (Eq. 13)
    ALPHA = 5.0
    BETA  = 5.0
    ZETA  = 0.5

    def __init__(
        self,
        episode_seed: int = 0,
        domain_randomize: bool = True,
        scenario: str = "normal",   # "normal"|"crit_load"|"pv_outage"|"dynamic_price"
    ):
        super().__init__()
        self.episode_seed     = episode_seed
        self.domain_randomize = domain_randomize
        self.scenario         = scenario

        # Spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self._build_profiles()

    # ------------------------------------------------------------------
    def _build_profiles(self):
        rng_seed = self.episode_seed if self.domain_randomize else 42
        self.pv_profile   = _solar_profile(self.T, seed=rng_seed) * 70.0   # kW peak ~70
        self.load_fix     = _load_profile(self.T, seed=rng_seed + 1) * 60.0 # kW
        self.load_flex    = _load_profile(self.T, seed=rng_seed + 2) * 20.0 # kW
        self.tariff       = _tou_tariff(self.T)
        self.ev_avail     = _ev_availability(self.T, seed=rng_seed + 3)

        # --- Scenario modifications ---
        if self.scenario == "crit_load":
            self.load_fix *= 1.30
        elif self.scenario == "pv_outage":
            self.pv_profile[6:12] = 0.0
        elif self.scenario == "dynamic_price":
            self.tariff *= np.random.default_rng(rng_seed).uniform(0.5, 2.0, self.T)
        elif self.scenario == "high_res":
            self.pv_profile *= 1.4

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.episode_seed = seed
        self._build_profiles()

        self.t        = 0
        self.soc_bess = 0.5
        self.soc_ev   = 0.6

        self.total_cost       = 0.0
        self.total_degradation = 0.0
        self.load_loss_count   = 0

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        t = min(self.t, self.T - 1)
        hour_angle = 2 * np.pi * t / self.T
        obs = np.array([
            2 * (self.soc_bess - self.SOC_BESS_MIN) / (self.SOC_BESS_MAX - self.SOC_BESS_MIN) - 1,
            2 * (self.soc_ev   - self.SOC_EV_MIN)   / (self.SOC_EV_MAX   - self.SOC_EV_MIN)   - 1,
            self.pv_profile[t] / 70.0 * 2 - 1,
            self.load_fix[t]   / 80.0 * 2 - 1,
            self.tariff[t]     / 0.25 * 2 - 1,
            np.sin(hour_angle),
            np.cos(hour_angle),
        ], dtype=np.float32).clip(-1, 1)
        return obs

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        t = self.t

        # --- De-normalise actions ---
        p_bess_cmd = float(action[0]) * self.P_BESS_MAX  # kW (+discharge, -charge)
        p_ev_cmd   = float(action[1]) * self.P_EV_MAX    # kW (+discharge, -charge)
        p_flex_cmd = (float(action[2]) + 1) / 2 * self.P_FLEX_MAX  # [0, P_FLEX_MAX]
        p_grid_cmd = float(action[3]) * self.P_GRID_MAX  # kW (+import, -export)

        # --- BESS update  (Eq. 2) ---
        if p_bess_cmd >= 0:   # discharge
            p_dis = min(p_bess_cmd, self.P_BESS_MAX)
            p_ch  = 0.0
        else:                 # charge
            p_ch  = min(-p_bess_cmd, self.P_BESS_MAX)
            p_dis = 0.0
        soc_new = (self.soc_bess
                   + (self.ETA_C_BESS * p_ch - p_dis / self.ETA_D_BESS)
                   / self.E_BESS_MAX * self.DT)
        soc_new = np.clip(soc_new, 0.0, 1.0)   # hard clamp before penalty
        p_bess_actual = p_dis - p_ch            # net dispatch

        # --- EV update  (Eq. 5) ---
        alpha = self.ev_avail[t]
        if alpha == 0:
            p_ev_actual = 0.0
            soc_ev_new  = self.soc_ev
        else:
            if p_ev_cmd >= 0:
                p_ev_dis = min(p_ev_cmd, self.P_EV_MAX)
                p_ev_ch  = 0.0
            else:
                p_ev_ch  = min(-p_ev_cmd, self.P_EV_MAX)
                p_ev_dis = 0.0
            soc_ev_new = (self.soc_ev
                          + alpha * (self.ETA_C_EV * p_ev_ch - p_ev_dis / self.ETA_D_EV)
                          / self.E_EV_MAX * self.DT)
            soc_ev_new  = np.clip(soc_ev_new, 0.0, 1.0)
            p_ev_actual = p_ev_dis - p_ev_ch

        # --- Flexible load ---
        p_flex_actual = np.clip(p_flex_cmd, self.P_FLEX_MIN, self.P_FLEX_MAX)
        p_flex_desired = self.load_flex[t]

        # --- Power balance  (Eq. 1) ---
        p_pv    = self.pv_profile[t]
        p_load  = self.load_fix[t] + p_flex_actual
        # grid fills the gap
        p_grid_needed = p_load - p_pv - p_bess_actual - p_ev_actual
        p_grid_actual = np.clip(p_grid_needed, self.P_GRID_MIN, self.P_GRID_MAX)

        # check load loss
        supply = p_pv + p_bess_actual + p_ev_actual + p_grid_actual
        load_loss = max(0.0, p_load - supply)
        if load_loss > 1.0:
            self.load_loss_count += 1

        # --- Rewards ---
        lam = self.tariff[t]

        # Cost  (Eq. 10)
        r_cost = -(lam * p_grid_actual * self.DT)

        # Battery degradation  (Eq. 11)
        p_bess_abs = abs(p_bess_actual)
        r_batt = -self.GAMMA * (p_bess_abs / self.P_BESS_MAX) ** self.KAPPA

        # Renewable utilisation  (Eq. 12)
        delta = 0.5
        r_env = delta * p_pv / max(p_load, 1e-3)

        # Stability / comfort  (Eq. 13)
        r_stab = (
            -self.ALPHA * max(0, self.SOC_BESS_MIN - soc_new)
            -self.BETA  * max(0, soc_new - self.SOC_BESS_MAX)
            -self.ZETA  * abs(p_flex_actual - p_flex_desired)
            -2.0        * load_loss
        )

        # Weighted total  (Eq. 9)
        reward = (
            self.W_C * r_cost
            + self.W_B * r_batt
            + self.W_E * r_env
            + self.W_S * r_stab
        )
        reward = float(np.clip(reward, -10.0, 10.0))

        # --- State updates ---
        self.soc_bess = float(soc_new)
        self.soc_ev   = float(soc_ev_new)
        self.total_cost       += lam * max(0, p_grid_actual) * self.DT
        self.total_degradation += abs(r_batt)
        self.t += 1

        terminated = self.t >= self.T
        info = {
            "t": t,
            "p_pv": p_pv,
            "p_bess": p_bess_actual,
            "p_ev": p_ev_actual,
            "p_grid": p_grid_actual,
            "p_load": p_load,
            "soc_bess": self.soc_bess,
            "soc_ev": self.soc_ev,
            "load_loss": load_loss,
            "reward": reward,
            "tariff": lam,
        }
        return self._get_obs(), reward, terminated, False, info
