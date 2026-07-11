"""
metrics.py
==========
Performance metrics defined in Section 3.2 of the paper.

  Eq. 15  – Total Energy Cost  (C_total)
  Eq. 16  – Voltage Deviation Index (VDI)
  Eq. 17  – Frequency Deviation Index (FDI)
  Eq. 19  – Battery Degradation  (D_batt)
  Eq. 20  – Robust Control Index for Resilience (RCIR)
  Eq. 21  – Loss of Load Probability (LOLP)
  Eq. 22  – Renewable Utilisation Ratio (RUR)
"""

import numpy as np
from typing import Sequence


# ---------------------------------------------------------------------------
# Eq. 15
# ---------------------------------------------------------------------------
def total_energy_cost(
    tariffs:   Sequence[float],
    p_grid:    Sequence[float],
    dt:        float = 1.0,
) -> float:
    """
    C_total = Σ λ(t) · P_Grid(t) · Δt

    Sign convention: P_Grid > 0 → import (cost), < 0 → export (revenue).
    """
    return float(sum(lam * p * dt for lam, p in zip(tariffs, p_grid)))


# ---------------------------------------------------------------------------
# Eq. 16 – 17
# ---------------------------------------------------------------------------
def voltage_deviation_index(
    voltages: Sequence[float],
    v_ref:    float = 1.0,
) -> float:
    """VDI = (1/T) Σ |V(t) - V_ref| / V_ref"""
    v = np.asarray(voltages)
    return float(np.mean(np.abs(v - v_ref) / v_ref))


def frequency_deviation_index(
    freqs:  Sequence[float],
    f_ref:  float = 50.0,
) -> float:
    """FDI = (1/T) Σ |f(t) - f_ref| / f_ref"""
    f = np.asarray(freqs)
    return float(np.mean(np.abs(f - f_ref) / f_ref))


# ---------------------------------------------------------------------------
# Eq. 19
# ---------------------------------------------------------------------------
def battery_degradation(soc_traj: list[float],
                        phi: float = 1.0,
                        kappa: float = 1.5,
                        soc_min: float = 0.1,   # FIX-19: matches env's real floor (was 0.2)
                        soc_max: float = 0.9) -> float:
    """
    D_batt = Σ φ · (ΔSOC_k / (SOC_max - SOC_min))^κ

    Cycle depth ΔSOC_k is computed by a simplified rainflow counting:
    each local minimum-to-maximum excursion is one half-cycle.
    """
    soc = np.asarray(soc_trajectory, dtype=float)
    span = soc_max - soc_min
    if span == 0:
        return 0.0

    # Simple peak/valley detection
    deltas = np.diff(soc)
    depth_sum = 0.0
    i = 0
    while i < len(deltas) - 1:
        # Find local min then local max (or vice versa)
        if deltas[i] * deltas[i + 1] < 0:          # direction change
            delta_soc = abs(soc[i + 1] - soc[i])
            depth_sum += phi * (delta_soc / span) ** kappa
        i += 1
    return float(depth_sum)


# ---------------------------------------------------------------------------
# Eq. 20
# ---------------------------------------------------------------------------
def rcir(costs: Sequence[float]) -> float:
    """
    RCIR = 1 - σ(C_total) / μ(C_total)

    Higher is better (less variance relative to mean).
    """
    c = np.asarray(costs, dtype=float)
    mu = c.mean()
    if mu <= 0:
        return 1.0
    return float(np.clip(1 - c.std() / mu, 0, 1))


# ---------------------------------------------------------------------------
# Eq. 21
# ---------------------------------------------------------------------------
def lolp(
    p_load:   Sequence[float],
    p_supply: Sequence[float],
) -> float:
    """
    LOLP = (1/T) Σ 1{P_Load(t) > P_Supply(t)}
    """
    load   = np.asarray(p_load)
    supply = np.asarray(p_supply)
    return float(np.mean(load > supply))


# ---------------------------------------------------------------------------
# Eq. 22
# ---------------------------------------------------------------------------
def renewable_utilisation_ratio(
    p_pv:   Sequence[float],
    p_wt:   Sequence[float],
    p_curt: Sequence[float],
    p_load: Sequence[float],
) -> float:
    """
    RUR = (Σ P_PV + Σ P_WT - Σ P_curt) / Σ P_Load
    """
    num = sum(p_pv) + sum(p_wt) - sum(p_curt)
    den = sum(p_load)
    return float(num / den) if den > 0 else 0.0


# ---------------------------------------------------------------------------
# Convenience: compute all metrics from a single episode log
# ---------------------------------------------------------------------------
def episode_metrics(log: dict) -> dict:
    """
    log keys expected:
        tariff, p_grid, p_pv, p_load, soc_bess, load_loss
    Optional:
        voltage, frequency, p_curt, p_wt
    """
    tariffs   = log["tariff"]
    p_grid    = log["p_grid"]
    p_pv      = log["p_pv"]
    p_load    = log["p_load"]
    soc_bess  = log["soc_bess"]
    load_loss = log["load_loss"]

    p_supply = [pl - ll for pl, ll in zip(p_load, load_loss)]
    p_curt   = log.get("p_curt", [0.0] * len(p_pv))
    p_wt     = log.get("p_wt",   [0.0] * len(p_pv))
    voltages  = log.get("voltage",   [1.0] * len(p_pv))
    freqs     = log.get("frequency", [50.0] * len(p_pv))

    return {
        "C_total": total_energy_cost(tariffs, p_grid),
        "VDI":     voltage_deviation_index(voltages),
        "FDI":     frequency_deviation_index(freqs),
        "D_batt":  battery_degradation(soc_bess),
        "LOLP":    lolp(p_load, p_supply),
        "RUR":     renewable_utilisation_ratio(p_pv, p_wt, p_curt, p_load),
    }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    T = 24
    rng = random.Random(42)

    fake_log = {
        "tariff":    [rng.uniform(0.08, 0.22) for _ in range(T)],
        "p_grid":    [rng.uniform(-10, 20)    for _ in range(T)],
        "p_pv":      [rng.uniform(0, 50)      for _ in range(T)],
        "p_load":    [rng.uniform(40, 80)     for _ in range(T)],
        "soc_bess":  [0.5 + 0.1 * np.sin(2 * np.pi * t / T) for t in range(T)],
        "load_loss": [max(0, rng.gauss(0, 1)) for _ in range(T)],
    }
    m = episode_metrics(fake_log)
    print("Metric check:")
    for k, v in m.items():
        print(f"  {k:10s} = {v:.4f}")

    # RCIR
    costs = [rng.uniform(50, 150) for _ in range(30)]
    print(f"\n  RCIR (30 runs) = {rcir(costs):.4f}")
