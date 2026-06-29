# microgrid_env.py  — v2 (fixed)
#
# FIXES vs v1:
#   FIX-1  PV solar profile peaks at noon (was peaking at 6am)
#   FIX-2  Load follows a realistic two-peak residential demand curve
#   FIX-3  Tariff uses a real TOU schedule (peak 09-21 = 0.20, off-peak = 0.08)
#   FIX-4  State is generated ONCE per step (stored in self._state) so the
#           obs the agent acted on, the physics, and next_obs are all consistent
#           (the original code called _get_obs() twice in step(), drawing
#            different random numbers each time — breaking MDP transitions)

import numpy as np
import gymnasium as gym
from microgrid_constants import DT, GAMMA, P_BESS_MAX, KAPPA, ZETA

CAP_EV = 40.0   # CHANGE-3: EV usable-energy normaliser (kWh-equivalent)


class MicrogridEnv(gym.Env):
    DT         = DT
    GAMMA      = GAMMA
    P_BESS_MAX = P_BESS_MAX
    KAPPA      = KAPPA
    ZETA       = ZETA
    T          = 24

    def __init__(self, episode_seed=0, domain_randomize=False, scenario="normal"):
        super().__init__()
        self.episode_seed     = episode_seed
        self.domain_randomize = domain_randomize
        self.scenario         = scenario

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self._t    = 0
        self._soc  = 0.5
        self._soc_ev = 0.5
        self._rng  = np.random.default_rng(0)
        self._state = self._blank_state()

    # ------------------------------------------------------------------
    def _blank_state(self):
        return dict(soc_bess=0.5, soc_ev=0.5, p_pv=0.0,
                    p_load=30.0, tariff=0.12, hour=0)

    # ------------------------------------------------------------------
    # FIX-4: generate the full state once and store it
    def _generate_state(self):
        """Draw a consistent state for the current timestep."""
        hour    = self._t % 24
        rng     = self._rng

        # FIX-1: solar arc peaks at noon
        pv_clear = max(0.0, np.sin(np.pi * (hour - 6) / 12))
        cloud    = rng.uniform(0.7, 1.0) if self.domain_randomize else 0.85
        p_pv     = float(pv_clear * cloud * 50.0)

        # FIX-2: residential two-peak load profile
        morning  = 0.3 * np.exp(-((hour - 8) ** 2) / 8.0)
        evening  = 0.5 * np.exp(-((hour - 19) ** 2) / 10.0)
        noise    = rng.normal(0, 0.04) if self.domain_randomize else 0.0
        load_norm = float(np.clip(0.35 + morning + evening + noise, 0.2, 1.0))
        p_load   = load_norm * 80.0

        # FIX-3: real TOU tariff
        base_tariff = 0.20 if (9 <= hour <= 21) else 0.08
        t_noise     = rng.normal(0, 0.01) if self.domain_randomize else 0.0
        tariff      = float(np.clip(base_tariff + t_noise, 0.06, 0.25))

        soc_ev = float(rng.uniform(0.2, 0.8)) if self.domain_randomize else 0.5

        self._state = dict(
            hour     = hour,
            soc_bess = self._soc,
            soc_ev   = soc_ev,
            p_pv     = p_pv,
            p_load   = p_load,
            tariff   = tariff,
        )

    # ------------------------------------------------------------------
    def _state_to_obs(self) -> np.ndarray:
        s    = self._state
        hour = s["hour"]
        return np.array([
            s["soc_bess"],
            s["soc_ev"],
            s["p_pv"] / 50.0,           # normalised 0-1
            s["p_load"] / 80.0,         # normalised 0-1
            s["tariff"],
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        self._rng = np.random.default_rng(self.episode_seed)
        self._t   = 0
        self._soc = float(self._rng.uniform(0.3, 0.7))
        self._soc_ev = float(self._rng.uniform(0.3, 0.7))   # CHANGE-3
        self._generate_state()          # generate state for t=0
        return self._state_to_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action, reward_weights=None):
        # --- reward weights ---
        if reward_weights is None:
            w_c, w_b, w_e, w_s = 1.0, 0.3, 0.2, 0.5
        else:
            rw = np.clip(reward_weights, 0.01, 10.0)
            w_c, w_b, w_e, w_s = float(rw[0]), float(rw[1]), float(rw[2]), float(rw[3])

        # --- current state (already generated, consistent with obs) ---
        s      = self._state
        tariff = s["tariff"]
        p_pv   = s["p_pv"]
        p_load = s["p_load"]

        # --- actions ---
        action = np.clip(action, -1, 1)
        p_bess = float(action[0]) * P_BESS_MAX
        p_ev   = float(action[1]) * 20.0
        p_flex = 30.0 + float(action[2]) * 10.0

        # CHANGE-1: limit BESS power to energy available at current SOC
        # (p_bess>0 = discharge lowers SOC; p_bess<0 = charge raises SOC)
        cap_dis =  (self._soc - 0.1) * (P_BESS_MAX * 2) / DT   # >=0
        cap_chg = -(0.9 - self._soc) * (P_BESS_MAX * 2) / DT   # <=0
        p_bess  = float(np.clip(p_bess, cap_chg, cap_dis))

        # CHANGE-3a: same coupling for the EV battery
        cap_dis_ev =  (self._soc_ev - 0.1) * CAP_EV / DT
        cap_chg_ev = -(0.9 - self._soc_ev) * CAP_EV / DT
        p_ev       = float(np.clip(p_ev, cap_chg_ev, cap_dis_ev))

        # --- power balance ---
        p_grid    = p_load - p_pv - p_bess - p_ev
        load_loss = max(0.0, p_grid - 50.0)   # CHANGE-2: unmet demand (import need > 50 cap)
        p_grid    = float(np.clip(p_grid, -50.0, 50.0))

        # --- SOC update ---
        self._soc = float(np.clip(
            self._soc - p_bess * DT / (P_BESS_MAX * 2), 0.1, 0.9
        ))

        # --- reward components ---
        r_cost    = -tariff * max(0.0, p_grid) * DT / 10.0
        r_battery = -GAMMA * (abs(p_bess) / P_BESS_MAX) ** KAPPA
        r_energy  = (p_pv / 50.0) * DT
        r_safe    = -load_loss / 50.0
        reward    = float(w_c*r_cost + w_b*r_battery + w_e*r_energy + w_s*r_safe)

        # --- advance time and pre-generate NEXT state ---
        self._t += 1
        done = self._t >= self.T
        if not done:
            self._generate_state()      # FIX-4: next_obs uses the new state
        else:
            self._state["soc_bess"] = self._soc

        next_obs = self._state_to_obs()

        info = {
            "tariff":    tariff,
            "p_grid":    p_grid,
            "p_bess":    p_bess,
            "p_ev":      p_ev,
            "p_flex":    p_flex,
            "p_load":    p_load,
            "p_pv":      p_pv,
            "soc_bess":  self._soc,
            "load_loss": load_loss,
        }
        return next_obs, reward, done, False, info
