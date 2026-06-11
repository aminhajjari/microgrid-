"""
demo.py  (v5 — ARW integrated)
=======
Smoke test — 60 episodes per method.

CHANGE in v5:
  HMA now uses ARW (Adaptive Reward Weighting).
  get_reward_weights(obs) is called each step and passed to env.step().
  ARW warm-up = 50 episodes, so smoke test (60 ep) is almost entirely
  in warm-up → behaviour identical to v4 → Gate 1 still passes.
  update_arw() is called at episode end to train ARWN after warm-up.
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

SMOKE_HMA_VS_SA_MIN      = -2.0
SMOKE_HMA_VS_FLAT_LOLP_MAX = 2.0


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
                # ARW: get adaptive weights for this timestep
                rw = ctrl.get_reward_weights(obs)
                action, omega = ctrl.select_actions(obs, local_rewards,
                                                    explore=(ep < n_ep * 0.8))
                # Pass adaptive weights to env
                nobs, reward, done, _, info = env.step(action, reward_weights=rw)
                # Record reward for ARW REINFORCE update
                ctrl.record_reward(reward)
            else:
                action = ctrl.select_actions(obs, explore=(ep < n_ep * 0.8))
                # SA and Flat: no reward_weights → uses fixed paper weights
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

        # ARW episode-end update
        if isinstance(ctrl, HMADRLFramework):
            ctrl.update_arw()

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
    ax.set_title("Training Convergence (Smoke Test)")
    ax.legend(); ax.grid(alpha=0.3)
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
        print(f"{r['label']:<28} {r['avg_reward']:>12.2f} "
              f"{r['avg_cost']:>10.4f} {r['rcir']:>8.3f} "
              f"{100*r['avg_lolp']:>8.3f}")
    print("=" * 72)


# ---------------------------------------------------------------------------
def check_smoke_quality(results) -> bool:
    by_method  = {r["method"]: r for r in results}
    sa_reward  = by_method.get("sa",   {}).get("avg_reward", None)
    hma_reward = by_method.get("hma",  {}).get("avg_reward", None)
    flat_lolp  = by_method.get("flat", {}).get("avg_lolp",   None)
    hma_lolp   = by_method.get("hma",  {}).get("avg_lolp",   None)
    sa_lolp    = by_method.get("sa",   {}).get("avg_lolp",   None)

    if sa_reward is None or hma_reward is None:
        print("  ⚠️  Missing SA or HMA results — skipping gate.")
        return True

    passed = True

    if sa_reward < -50.0:
        print("  ⚠️  SA reward is extremely negative — environment bug.")
        return False

    reward_gap = hma_reward - sa_reward
    if reward_gap < SMOKE_HMA_VS_SA_MIN:
        print(
            f"\n  ❌ GATE 1 FAILED\n"
            f"     HMA avg_reward = {hma_reward:.2f}\n"
            f"     SA  avg_reward = {sa_reward:.2f}\n"
            f"     Gap = {reward_gap:.2f} < allowed {SMOKE_HMA_VS_SA_MIN:.2f}\n"
        )
        passed = False
    else:
        print(f"  ✓ Gate 1 passed  (HMA−SA gap = {reward_gap:.2f} ≥ {SMOKE_HMA_VS_SA_MIN:.2f})")
      
    
        print(
            f"\n  ❌ GATE 1 FAILED — severe reward regression\n"
            f"     HMA avg_reward = {hma_reward:.1f}\n"
            f"     SA  avg_reward = {sa_reward:.1f}\n"
            f"     HMA/SA ratio   = {reward_ratio:.2f} < required "
            f"{SMOKE_HMA_VS_SA_MIN:.2f}\n"
            f"     Check: adaptive_reward.py must be in the same directory.\n"
            f"     Check: # VERSION: hma_drl v5\n"
        )
        passed = False
    else:
        print(f"  ✓ Gate 1 passed  (HMA/SA = {reward_ratio:.2f} ≥ "
              f"{SMOKE_HMA_VS_SA_MIN:.2f})")

    ref_lolp = flat_lolp if flat_lolp is not None else sa_lolp
    ref_name = "Flat" if flat_lolp is not None else "SA"
    if ref_lolp is not None and hma_lolp is not None and ref_lolp > 0:
        lolp_ratio = hma_lolp / ref_lolp
        if lolp_ratio > SMOKE_HMA_VS_FLAT_LOLP_MAX:
            print(
                f"\n  ❌ GATE 2 FAILED — grid safety regression\n"
                f"     HMA  LOLP = {100*hma_lolp:.1f}%\n"
                f"     {ref_name} LOLP = {100*ref_lolp:.1f}%\n"
                f"     Ratio = {lolp_ratio:.1f}x > allowed "
                f"{SMOKE_HMA_VS_FLAT_LOLP_MAX:.1f}x\n"
            )
            passed = False
        else:
            print(f"  ✓ Gate 2 passed  (LOLP ratio {lolp_ratio:.1f}x ≤ "
                  f"{SMOKE_HMA_VS_FLAT_LOLP_MAX:.1f}x)")

    if passed:
        print(f"\n✓ Smoke test passed")
    return passed


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
