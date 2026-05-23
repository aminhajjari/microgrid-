"""
demo.py  (FIXED v2)
=======
Quick end-to-end smoke test (60 episodes per method).

==========================================================================
CHANGE in v2  (FIX-S2)
==========================================================================

FIX-S2  Smoke test now compares HMA vs FLAT, not HMA vs SA.

        Why the old threshold was wrong:
          - SA converges fast: 1 agent, 1 unified reward, no coordination
            overhead. It reaches ~95 reward within 60 episodes.
          - HMA has 5 agents. The supervisor needs 2000 transitions before
            it trains at all. 60 episodes × 24 steps = 1440 transitions,
            so the supervisor NEVER trains during the smoke test.
          - Therefore HMA ≈ Flat MA-DRL at this episode count.
            Comparing HMA to SA at 60 episodes always fails by design.

        The correct comparison at equal episode count is HMA vs Flat:
          - They have the same local agent architecture.
          - The only difference is HMA adds omega modulation on BESS/EV.
          - HMA should reach at least 90% of Flat reward even with random omega.
          - If HMA falls below 90% of Flat, the omega modulation or local
            reward decomposition is actively hurting performance.

        Validation against both runs:
          Original broken run: HMA=36.2, Flat=45.7 → ratio=0.79 → FAIL ✓
          New fixed run:       HMA=41.2, Flat=43.8 → ratio=0.94 → PASS ✓
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from microgrid_env import MicrogridEnv
from hma_drl import HMADRLFramework, FlatMADRL, SingleAgentDRL
from metrics import rcir, lolp


N_EPISODES = 60
BATCH_SIZE = 128
DEVICE     = "cpu"
OUT_DIR    = Path("/home/gkianfar/scratch/Amin/MSH/output/plots")
OUT_DIR.mkdir(exist_ok=True)

# FIX-S2: compare HMA vs Flat (correct baseline at equal episode count)
# HMA must reach at least this fraction of Flat reward
SMOKE_HMA_VS_FLAT_MIN = 0.90


# ---------------------------------------------------------------------------
def run_one(method: str, n_ep: int = N_EPISODES):
    env = MicrogridEnv(episode_seed=0, domain_randomize=True)

    if method == "hma":
        ctrl  = HMADRLFramework(device=DEVICE)
        label = "Hierarchical MA-DRL"
    elif method == "flat":
        ctrl  = FlatMADRL(device=DEVICE)
        label = "Flat MA-DRL"
    else:
        ctrl  = SingleAgentDRL(device=DEVICE)
        label = "Single-Agent DRL"

    rewards, costs, lolps_ep, soc_trajs = [], [], [], []
    local_rewards = np.zeros(4)

    for ep in range(n_ep):
        env.episode_seed = ep
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_cost   = 0.0
        soc_ep    = []
        p_load_ep, p_sup_ep = [], []

        for _ in range(env.T):
            if isinstance(ctrl, HMADRLFramework):
                action, omega = ctrl.select_actions(obs, local_rewards,
                                                    explore=(ep < n_ep * 0.8))
            else:
                action = ctrl.select_actions(obs, explore=(ep < n_ep * 0.8))

            nobs, reward, done, _, info = env.step(action)

            if isinstance(ctrl, HMADRLFramework):
                prev          = local_rewards.copy()
                local_rewards = ctrl.compute_local_rewards(info)
                ctrl.store_transitions(obs, action, local_rewards,
                                       reward, nobs, done, prev, omega)
            elif isinstance(ctrl, FlatMADRL):
                ctrl.store_transitions(obs, action,
                                       np.full(4, reward / 4), nobs, done)
            else:
                ctrl.store_transitions(obs, action, reward, nobs, done)

            ctrl.update_all(BATCH_SIZE)

            ep_reward += reward
            ep_cost   += info["tariff"] * max(0, info["p_grid"])
            soc_ep.append(info["soc_bess"])
            p_load_ep.append(info["p_load"])
            p_sup_ep.append(info["p_load"] - info["load_loss"])
            obs = nobs

        rewards.append(ep_reward)
        costs.append(ep_cost)
        lolps_ep.append(lolp(p_load_ep, p_sup_ep))
        soc_trajs.append(soc_ep)

    return {
        "method":     method,
        "label":      label,
        "rewards":    rewards,
        "costs":      costs,
        "lolps":      lolps_ep,
        "soc_last":   soc_trajs[-1],
        "rcir":       rcir(costs[-20:]),
        "avg_cost":   float(np.mean(costs[-20:])),
        "avg_lolp":   float(np.mean(lolps_ep[-20:])),
        "avg_reward": float(np.mean(rewards[-20:])),
    }


# ---------------------------------------------------------------------------
def make_plots(results):
    colors = {"hma": "#E84855", "flat": "#FF9F1C", "sa": "#2EC4B6"}

    fig, ax = plt.subplots(figsize=(9, 4))
    for r in results:
        rw = np.array(r["rewards"])
        sm = np.convolve(rw, np.ones(5) / 5, mode="valid")
        ax.plot(sm, color=colors[r["method"]], lw=2, label=r["label"])
    ax.set_xlabel("Episode"); ax.set_ylabel("Episodic Reward")
    ax.set_title("Training Convergence"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(OUT_DIR / "convergence.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    base = max(r["avg_cost"] for r in results) + 1e-9
    for r in results:
        sav = max(0, 100 * (1 - r["avg_cost"] / base))
        ax.bar(r["label"], sav, color=colors[r["method"]], alpha=0.85)
    ax.set_ylabel("Cost Reduction (%)"); ax.set_title("Average Cost Reduction")
    ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    fig.savefig(OUT_DIR / "cost_reduction.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(9, 4))
    for r in results:
        ax.plot(r["soc_last"], color=colors[r["method"]], lw=2, label=r["label"])
    ax.set_xlabel("Time (h)"); ax.set_ylabel("SOC (p.u.)")
    ax.set_title("SOC Trajectories – Last Episode")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(OUT_DIR / "soc_trajectories.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    data = [r["rewards"][-20:] for r in results]
    lbls = [r["label"] for r in results]
    bp   = ax.boxplot(data, tick_labels=lbls, patch_artist=True,
                      medianprops={"color": "black", "lw": 2})
    for patch, r in zip(bp["boxes"], results):
        patch.set_facecolor(colors[r["method"]]); patch.set_alpha(0.7)
    ax.set_ylabel("Reward (last 20 episodes)")
    ax.set_title("Reward Variability"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); fig.savefig(OUT_DIR / "reward_variability.png", dpi=150); plt.close()

    print(f"Plots saved → {OUT_DIR}/")


# ---------------------------------------------------------------------------
def print_table(results):
    print()
    print("=" * 72)
    print(f"{'Method':<28} {'Avg Reward':>12} {'Avg Cost':>10} "
          f"{'RCIR':>8} {'LOLP%':>8}")
    print("=" * 72)
    for r in results:
        print(
            f"{r['label']:<28} "
            f"{r['avg_reward']:>12.2f} "
            f"{r['avg_cost']:>10.4f} "
            f"{r['rcir']:>8.3f} "
            f"{100*r['avg_lolp']:>8.3f}"
        )
    print("=" * 72)


# ---------------------------------------------------------------------------
# FIX-S2: quality gate — HMA vs Flat
# ---------------------------------------------------------------------------
def check_smoke_quality(results) -> bool:
    by_method  = {r["method"]: r for r in results}
    flat_reward = by_method.get("flat", {}).get("avg_reward", None)
    hma_reward  = by_method.get("hma",  {}).get("avg_reward", None)
    sa_reward   = by_method.get("sa",   {}).get("avg_reward", None)

    if flat_reward is None or hma_reward is None:
        print("  ⚠️  Could not find both HMA and Flat results — skipping gate.")
        return True

    # Sanity check: warn if SA also failed to converge
    if sa_reward is not None and sa_reward < 50:
        print(f"  ⚠️  SA reward is low ({sa_reward:.1f}) — environment may have a bug.")

    # Primary gate: HMA must not regress significantly vs Flat
    if flat_reward <= 0:
        print("  ⚠️  Flat reward is non-positive — environment may have a bug.")
        return False

    ratio = hma_reward / flat_reward
    if ratio < SMOKE_HMA_VS_FLAT_MIN:
        print(
            f"\n  ❌ SMOKE TEST FAILED\n"
            f"     HMA  avg_reward = {hma_reward:.1f}\n"
            f"     Flat avg_reward = {flat_reward:.1f}\n"
            f"     HMA/Flat ratio  = {ratio:.2f} < required {SMOKE_HMA_VS_FLAT_MIN:.2f}\n"
            f"\n"
            f"     At {N_EPISODES} episodes the supervisor has not yet trained\n"
            f"     (needs 2000 transitions; smoke test generates only\n"
            f"     {N_EPISODES}×24={N_EPISODES*24} steps). HMA and Flat should\n"
            f"     therefore be roughly equal. A large gap means the omega\n"
            f"     modulation or local reward decomposition is actively\n"
            f"     hurting performance. Check hma_drl.py before resubmitting.\n"
        )
        return False

    # Secondary advisory: HMA LOLP should not be much worse than Flat
    hma_lolp  = by_method["hma"].get("avg_lolp", 0)
    flat_lolp = by_method["flat"].get("avg_lolp", 0)
    if flat_lolp > 0 and hma_lolp > flat_lolp * 1.5:
        print(
            f"  ⚠️  Advisory: HMA LOLP ({100*hma_lolp:.1f}%) is >1.5× Flat LOLP "
            f"({100*flat_lolp:.1f}%).\n"
            f"     This suggests omega modulation is restricting grid import.\n"
            f"     Training will continue, but check FIX-F in hma_drl.py.\n"
        )

    print(f"✓ Smoke test passed  (HMA/Flat = {ratio:.2f} ≥ {SMOKE_HMA_VS_FLAT_MIN:.2f})")
    return True


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Running demo with {N_EPISODES} episodes per method …\n")
    results = []
    for m in ["sa", "flat", "hma"]:
        print(f"  → {m.upper()} …", end="", flush=True)
        r = run_one(m)
        results.append(r)
        print(f"  done. avg_reward={r['avg_reward']:.1f}")

    print_table(results)
    make_plots(results)
    print("\nAll done. Check plots/ for figures.")

    if not check_smoke_quality(results):
        sys.exit(1)
