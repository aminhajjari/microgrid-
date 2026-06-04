# microgrid_env.py
import numpy as np
import gymnasium as gym
from microgrid_constants import DT, GAMMA, P_BESS_MAX, KAPPA, ZETA

class MicrogridEnv(gym.Env):
    DT        = DT
    GAMMA     = GAMMA
    P_BESS_MAX = P_BESS_MAX
    KAPPA     = KAPPA
    ZETA      = ZETA

    T = 24  # timesteps per episode

    def __init__(self, episode_seed=0, domain_randomize=False, scenario="normal"):
        super().__init__()
        self.episode_seed      = episode_seed
        self.domain_randomize  = domain_randomize
        self.scenario          = scenario

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self._t   = 0
        self._soc = 0.5

    def reset(self, seed=None, options=None):
        rng = np.random.default_rng(self.episode_seed)
        self._t   = 0
        self._soc = rng.uniform(0.3, 0.7)
        self._rng = rng
        return self._get_obs(), {}

    def _get_obs(self):
        # [soc_bess, soc_ev, p_pv_norm, p_load_norm, tariff, hour_sin, hour_cos]
        hour = self._t % 24
        rng  = self._rng
        return np.array([
            self._soc,
            rng.uniform(0.2, 0.8),
            max(0, np.sin(np.pi * hour / 12)) * rng.uniform(0.8, 1.2),
            rng.uniform(0.4, 1.0),
            rng.uniform(0.08, 0.22),
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
        ], dtype=np.float32)

    def step(self, action, reward_weights=None):
        if reward_weights is None:
            w_c, w_b, w_e, w_s = 1.0, 0.3, 0.2, 0.5
        else:
            rw = np.clip(reward_weights, 0.01, 10.0)
            w_c, w_b, w_e, w_s = rw[0], rw[1], rw[2], rw[3]

        action = np.clip(action, -1, 1)
        p_bess = float(action[0]) * P_BESS_MAX
        p_ev   = float(action[1]) * 20.0
        p_flex = 30.0 + float(action[2]) * 10.0

        # Use current timestep for physics — call _get_obs() ONCE here
        obs    = self._get_obs()
        tariff = float(obs[4])
        p_pv   = float(obs[2]) * 50.0
        p_load = float(obs[3]) * 80.0

        p_grid    = p_load - p_pv - p_bess - p_ev
        load_loss = max(0.0, -p_grid - 50.0)
        p_grid    = np.clip(p_grid, -50.0, 50.0)

        self._soc = np.clip(
            self._soc - p_bess * DT / (P_BESS_MAX * 2), 0.1, 0.9
        )

        r_cost    = -tariff * max(0, p_grid) * DT * 0.1      # scale down cost penalty
        r_battery = -GAMMA * (abs(p_bess) / P_BESS_MAX) ** KAPPA
        r_energy  = p_pv * DT * 0.05                          # increase renewable bonus
        r_safe    = -load_loss * 0.1                           # scale down loss penalty
        reward    = w_c*r_cost + w_b*r_battery + w_e*r_energy + w_s*r_safe

        self._t += 1
        done = self._t >= self.T

        info = {
            "tariff": tariff, "p_grid": p_grid, "p_bess": p_bess,
            "p_ev": p_ev, "p_flex": p_flex, "p_load": p_load,
            "p_pv": p_pv, "soc_bess": self._soc, "load_loss": load_loss,
        }

        # Increment happened above, so next call to _get_obs() gives next timestep
        next_obs = self._get_obs()
        return next_obs, float(reward), done, False, info
