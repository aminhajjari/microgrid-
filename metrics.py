"""
metrics.py — merged (was metrics.py + metrics_v2.py)

Canonical metrics module: everything train.py / demo.py import comes from
here now. Previously the codebase had two files — metrics.py (paper
Eq.15-22 labeled, but unused and containing a NameError bug in
battery_degradation) and metrics_v2.py (actually imported everywhere,
missing Eq.15-17). Merged into one file so there is exactly one
implementation of each metric.

  rcir()                    — Eq. 20, Robust Control Index for Resilience
  lolp()                    — Eq. 21, Loss of Load Probability
  battery_degradation()     — Eq. 19, SOC cycle-depth degradation (canonical
                               impl — the old metrics.py version referenced
                               an undefined `soc_trajectory` variable and
                               would have crashed if ever called)
  degradation()             — Eq. 19 wrapper, same formula, Table-4-facing name
  rur()                     — Eq. 22, Renewable Utilization Ratio
  convergence_ep()          — episode at which 95% of final avg reward is reached
  total_energy_cost()       — Eq. 15 (not currently fed by microgrid_env.py —
                               kept for future use, see note below)
  voltage_deviation_index() — Eq. 16 (NOT currently usable — see note below)
  frequency_deviation_index()-Eq. 17 (NOT currently usable — see note below)
  EpisodeTimer, build_table4_row(), print_table4() — Table 4 reporting helpers

NOTE on voltage/frequency/power-loss metrics: microgrid_env.py models a
single aggregated node — there is no bus topology, admittance matrix, or
power-flow solver, so no per-bus voltage, frequency, or network power-loss
signal is ever produced. voltage_deviation_index() and
frequency_deviation_index() are kept here (correct, standalone formulas)
for if/when a multi-bus model is added, but nothing in this codebase
currently calls them with real data — do not report VDI/FDI/bus-voltage
results in the paper unless a power-flow model is actually added upstream.
"""

from __future__ import annotations
import numpy as np
import time
from typing import Sequence


# ──────────────────────────────────────────────────────────────────────────────
# Eq. 20 — Robust Control Index for Resilience
# ──────────────────────────────────────────────────────────────────────────────
def rcir(costs: list[float]) -> float:
    """RCIR = 1 - std(C) / mean(C). Returns 1.0 if all costs identical."""
    arr = np.array(costs, dtype=np.float64)
    mu  = np.mean(arr)
    if abs(mu) < 1e-9:
        return 1.0
    return float(1.0 - np.std(arr) / abs(mu))


# ──────────────────────────────────────────────────────────────────────────────
# Eq. 21 — Loss of Load Probability
# ──────────────────────────────────────────────────────────────────────────────
def lolp(p_load: list[float], p_supply: list[float]) -> float:
    """LOLP = fraction of timesteps where p_load > p_supply."""
    pl = np.array(p_load,   dtype=np.float64)
    ps = np.array(p_supply, dtype=np.float64)
    if len(pl) == 0:
        return 0.0
    return float(np.mean(pl > ps))


# ──────────────────────────────────────────────────────────────────────────────
# Eq. 19 — SOC cycle-depth degradation (canonical implementation)
# ──────────────────────────────────────────────────────────────────────────────
def battery_degradation(soc_traj: list[float],
                        phi: float = 1.0,
                        kappa: float = 1.5,
                        soc_min: float = 0.1,   # matches microgrid_env.py's real floor
                        soc_max: float = 0.9) -> float:
    """
    D_batt = sum_k phi * (|dSOC_k| / (SOC_max - SOC_min)) ^ kappa
    """
    soc  = np.array(soc_traj, dtype=np.float64)
    diff = np.abs(np.diff(soc))
    rng  = max(soc_max - soc_min, 1e-6)
    return float(np.sum(phi * (diff / rng) ** kappa))


def degradation(soc_traj: list[float],
                phi: float = 1.0,
                kappa: float = 1.5,
                soc_min: float = 0.1,
                soc_max: float = 0.9) -> float:
    """Same formula as battery_degradation() — Table-4-facing alias."""
    return battery_degradation(soc_traj, phi=phi, kappa=kappa,
                               soc_min=soc_min, soc_max=soc_max)


# ──────────────────────────────────────────────────────────────────────────────
# Eq. 22 — Renewable Utilization Ratio
# ──────────────────────────────────────────────────────────────────────────────
def rur(p_pv: list[float],
        p_load: list[float],
        p_curt: list[float] | None = None,
        p_wt: list[float] | None = None) -> float:
    """RUR = sum(P_PV + P_WT - P_curt) / sum(P_load)."""
    pv   = np.array(p_pv,   dtype=np.float64)
    load = np.array(p_load, dtype=np.float64)
    curt = np.zeros_like(pv) if p_curt is None else np.array(p_curt, dtype=np.float64)
    wt   = np.zeros_like(pv) if p_wt   is None else np.array(p_wt,   dtype=np.float64)

    total_load = np.sum(load)
    if total_load < 1e-9:
        return 0.0
    return float(np.sum(pv + wt - curt) / total_load)


# ──────────────────────────────────────────────────────────────────────────────
# Training-convergence helper
# ──────────────────────────────────────────────────────────────────────────────
def convergence_ep(rewards: list[float],
                   threshold: float = 0.95,
                   window: int = 50) -> int:
    """First episode at which the smoothed reward reaches `threshold` fraction
    of the final steady-state average. Returns len(rewards) if never reached."""
    arr = np.array(rewards, dtype=np.float64)
    if len(arr) < window:
        return len(arr)

    kernel = np.ones(window) / window
    smooth = np.convolve(arr, kernel, mode='valid')
    target = threshold * float(np.mean(arr[-window:]))

    for i, val in enumerate(smooth):
        if val >= target:
            return int(i + window)
    return len(arr)


# ──────────────────────────────────────────────────────────────────────────────
# Eq. 15-17 — kept for future use; NOT fed real data by microgrid_env.py today
# ──────────────────────────────────────────────────────────────────────────────
def total_energy_cost(tariffs: Sequence[float],
                      p_grid: Sequence[float],
                      dt: float = 1.0) -> float:
    """C_total = sum(lambda(t) * P_Grid(t) * dt). P_Grid>0=import(cost)."""
    return float(sum(lam * p * dt for lam, p in zip(tariffs, p_grid)))


def voltage_deviation_index(voltages: Sequence[float], v_ref: float = 1.0) -> float:
    """VDI = mean(|V(t) - V_ref| / V_ref). Requires per-node voltage data
    that microgrid_env.py does not currently produce (single-node model)."""
    v = np.asarray(voltages)
    return float(np.mean(np.abs(v - v_ref) / v_ref))


def frequency_deviation_index(freqs: Sequence[float], f_ref: float = 50.0) -> float:
    """FDI = mean(|f(t) - f_ref| / f_ref). Requires frequency data that
    microgrid_env.py does not currently produce."""
    f = np.asarray(freqs)
    return float(np.mean(np.abs(f - f_ref) / f_ref))


# ──────────────────────────────────────────────────────────────────────────────
# Episode timer helper
# ──────────────────────────────────────────────────────────────────────────────
class EpisodeTimer:
    """Wall-clock timer for 'Execution Time (s/episode)'."""

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
# Table 4 reporting helpers
# ──────────────────────────────────────────────────────────────────────────────
def build_table4_row(method: str,
                     result: dict,
                     baseline_cost: float,
                     baseline_deg:  float,
                     timer: EpisodeTimer) -> dict:
    """Compute all Table 4 columns for one method and return as a dict."""
    avg_cost   = result["avg_cost"]
    cost_red   = max(0.0, 100.0 * (1.0 - avg_cost / baseline_cost)) if baseline_cost > 0 else 0.0

    rur_vals = [
        rur(result["p_pv_ep"][i], result["p_load_ep"][i],
            result.get("p_curt_ep", [None]*len(result["p_pv_ep"]))[i])
        for i in range(len(result["p_pv_ep"]))
    ]
    avg_rur = float(np.mean(rur_vals[-20:])) if len(rur_vals) >= 20 else float(np.mean(rur_vals))

    avg_rcir = rcir(result["costs"][-20:])
    conv_ep  = convergence_ep(result["rewards"])

    deg_vals = [degradation(traj) for traj in result["soc_trajs"]]
    avg_deg  = float(np.mean(deg_vals[-20:]))
    rel_deg  = avg_deg / baseline_deg if baseline_deg > 0 else 1.0

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
