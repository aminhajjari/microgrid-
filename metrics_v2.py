"""
metrics.py  — v2 (paper Table 4 + Table 6 compatible)

NEW in v2:
  rur()           — Renewable Utilization Ratio (Eq. 22)
  degradation()   — SOC cycle-depth degradation index (Eq. 19)
  convergence_ep()— episode at which 95% of final avg reward is reached
  All existing functions (rcir, lolp, battery_degradation) unchanged.

These additions let demo.py and train.py produce every column in
paper Table 4 and Table 6 without changing existing metric calls.
"""

from __future__ import annotations
import numpy as np
import time


# ──────────────────────────────────────────────────────────────────────────────
# EXISTING (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def rcir(costs: list[float]) -> float:
    """
    Robust Control Index for Resilience (Eq. 20).
    RCIR = 1 - std(C) / mean(C).  Returns 1.0 if all costs identical.
    """
    arr = np.array(costs, dtype=np.float64)
    mu  = np.mean(arr)
    if abs(mu) < 1e-9:
        return 1.0
    return float(1.0 - np.std(arr) / abs(mu))


def lolp(p_load: list[float], p_supply: list[float]) -> float:
    """
    Loss of Load Probability (Eq. 21).
    = fraction of timesteps where p_load > p_supply.
    """
    pl = np.array(p_load,   dtype=np.float64)
    ps = np.array(p_supply, dtype=np.float64)
    if len(pl) == 0:
        return 0.0
    return float(np.mean(pl > ps))


def battery_degradation(soc_traj: list[float],
                        phi: float = 1.0,
                        kappa: float = 1.5,
                        soc_min: float = 0.2,
                        soc_max: float = 0.9) -> float:
    """
    Existing rainflow-lite degradation (kept for backward compat).
    """
    soc  = np.array(soc_traj, dtype=np.float64)
    diff = np.abs(np.diff(soc))
    rng  = max(soc_max - soc_min, 1e-6)
    return float(np.sum(phi * (diff / rng) ** kappa))


# ──────────────────────────────────────────────────────────────────────────────
# NEW — paper Table 4 metrics
# ──────────────────────────────────────────────────────────────────────────────

def rur(p_pv: list[float],
        p_load: list[float],
        p_curt: list[float] | None = None,
        p_wt: list[float] | None = None) -> float:
    """
    Renewable Utilization Ratio (Eq. 22).

        RUR = sum(P_PV + P_WT - P_curt) / sum(P_load)

    Parameters
    ----------
    p_pv   : PV generation at each timestep  (kW)
    p_load : Total demand at each timestep   (kW)
    p_curt : Curtailed renewable power       (kW); zeros if None
    p_wt   : Wind generation                 (kW); zeros if None

    Returns
    -------
    float in [0, 1+]  — higher is better
    """
    pv   = np.array(p_pv,   dtype=np.float64)
    load = np.array(p_load, dtype=np.float64)
    curt = np.zeros_like(pv) if p_curt is None else np.array(p_curt, dtype=np.float64)
    wt   = np.zeros_like(pv) if p_wt   is None else np.array(p_wt,   dtype=np.float64)

    total_load = np.sum(load)
    if total_load < 1e-9:
        return 0.0
    return float(np.sum(pv + wt - curt) / total_load)


def degradation(soc_traj: list[float],
                phi: float = 1.0,
                kappa: float = 1.5,
                soc_min: float = 0.1,   # FIX-19: matches env's real floor (was 0.2)
                soc_max: float = 0.9) -> float:
    """
    SOC cycle-depth degradation index (Eq. 19).

        D_batt = sum_{k} phi * (dSOC_k / (SOC_max - SOC_min))^kappa

    This is the same formula as battery_degradation() — kept as a
    separate name so Table 4 code reads clearly as `degradation(soc)`.
    Normalise against SA-DRL baseline in the caller to get the
    'relative' index from Table 4 (SA=1.0, Flat≈0.88, HMA≈0.75).
    """
    return battery_degradation(soc_traj, phi=phi, kappa=kappa,
                               soc_min=soc_min, soc_max=soc_max)


def convergence_ep(rewards: list[float],
                   threshold: float = 0.95,
                   window: int = 50) -> int:
    """
    First episode at which the smoothed reward reaches `threshold`
    fraction of the final steady-state average.

    Used for the 'Training Convergence (episodes)' column in Table 4.

    Parameters
    ----------
    rewards   : list of per-episode rewards across full training run
    threshold : fraction of final avg to consider 'converged' (paper uses 95%)
    window    : smoothing window for reward curve

    Returns
    -------
    int — episode index (1-based); returns len(rewards) if never reached
    """
    arr = np.array(rewards, dtype=np.float64)
    if len(arr) < window:
        return len(arr)

    # smooth with a moving average
    kernel = np.ones(window) / window
    smooth = np.convolve(arr, kernel, mode='valid')

    # final steady-state = mean of last `window` episodes
    target = threshold * float(np.mean(arr[-window:]))

    for i, val in enumerate(smooth):
        if val >= target:
            return int(i + window)            # 1-based episode number
    return len(arr)


# ──────────────────────────────────────────────────────────────────────────────
# NEW — episode timer helper
# ──────────────────────────────────────────────────────────────────────────────

class EpisodeTimer:
    """
    Lightweight wall-clock timer for 'Execution Time (s/episode)'.

    Usage:
        timer = EpisodeTimer()
        for ep in range(N):
            timer.start()
            # ... run episode ...
            timer.stop()
        print(timer.mean())   # → avg seconds/episode
    """

    def __init__(self):
        self._times: list[float] = []
        self._t0: float | None   = None

    def start(self):
        self._t0 = time.perf_counter()

    def stop(self):
        if self._t0 is not None:
            self._times.append(time.perf_counter() - self._t0)
            self._t0 = None

    def mean(self) -> float:
        return float(np.mean(self._times)) if self._times else 0.0

    def reset(self):
        self._times.clear()
        self._t0 = None


# ──────────────────────────────────────────────────────────────────────────────
# NEW — build_table4_row()  convenience function for demo.py / train.py
# ──────────────────────────────────────────────────────────────────────────────

def build_table4_row(method: str,
                     result: dict,
                     baseline_cost: float,
                     baseline_deg:  float,
                     timer: EpisodeTimer) -> dict:
    """
    Compute all Table 4 columns for one method and return as a dict.

    Parameters
    ----------
    method        : "sa" | "flat" | "hma"
    result        : dict returned by run_one() in demo.py
                    Must include keys: rewards, costs, lolps, soc_trajs,
                    p_pv_ep, p_load_ep, p_curt_ep (see demo.py v6 additions)
    baseline_cost : avg_cost of SA-DRL (for cost_reduction %)
    baseline_deg  : degradation() of SA-DRL (for relative index)
    timer         : EpisodeTimer that was running during run_one()

    Returns
    -------
    dict with keys matching Table 4 columns
    """
    avg_cost   = result["avg_cost"]
    cost_red   = max(0.0, 100.0 * (1.0 - avg_cost / baseline_cost)) if baseline_cost > 0 else 0.0

    # RUR: average over all episodes, using last-20 for consistency
    rur_vals = [
        rur(result["p_pv_ep"][i], result["p_load_ep"][i], result.get("p_curt_ep", [None]*len(result["p_pv_ep"]))[i])
        for i in range(len(result["p_pv_ep"]))
    ]
    avg_rur = float(np.mean(rur_vals[-20:])) if len(rur_vals) >= 20 else float(np.mean(rur_vals))

    # RCIR
    avg_rcir = rcir(result["costs"][-20:])

    # Training convergence
    conv_ep = convergence_ep(result["rewards"])

    # SOC degradation index (relative to SA baseline)
    deg_vals = [degradation(traj) for traj in result["soc_trajs"]]
    avg_deg  = float(np.mean(deg_vals[-20:]))
    rel_deg  = avg_deg / baseline_deg if baseline_deg > 0 else 1.0

    # Execution time
    exec_time = timer.mean()

    return {
        "Method":                         method.upper(),
        "Avg Cost Reduction (%)":         round(cost_red, 1),
        "RUR":                            round(avg_rur,  2),
        "RCIR":                           round(avg_rcir, 2),
        "Training Convergence (episodes)": conv_ep,
        "SOC Degradation Index (relative)": round(rel_deg, 2),
        "Execution Time (s/episode)":      round(exec_time, 3),
    }


def print_table4(rows: list[dict]):
    """Pretty-print Table 4 to stdout."""
    cols = ["Method", "Avg Cost Reduction (%)", "RUR", "RCIR",
            "Training Convergence (episodes)", "SOC Degradation Index (relative)",
            "Execution Time (s/episode)"]
    widths = [28, 22, 6, 6, 32, 30, 22]
    header = "".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    sep    = "-" * len(header)
    print()
    print("TABLE 4 — Comparative Results")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        line = "".join(f"{str(r.get(c,'')):<{w}}" for c, w in zip(cols, widths))
        print(line)
    print(sep)
