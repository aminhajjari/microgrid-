"""
demo.py
=======
Quick end-to-end smoke test (50 episodes per method).
Prints a comparison table and saves 4 plots to ./demo_plots/.

Run:
    python demo.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from microgrid_env import MicrogridEnv
from hma_drl import HMADRLFramework, FlatMADRL, SingleAgentDRL
from metrics import rcir, lolp, renewable_utilisation_ratio, battery_degradation


N_EPISODES = 60   # small enough to run in seconds; increase for real training
BATCH_SIZE = 128
DEVICE = "cpu"
OUT_DIR = Path("/home/gkianfar/scratch/Amin/MSH/output/plots")
OUT_DIR.mkdir(exist_ok=True)


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
                action = ctrl.select_actions(obs, local_rewards, explore=(ep < n_ep * 0.8))
            elif isinstance(ctrl, FlatMADRL):
                action = ctrl.select_actions(obs, explore=(ep < n_ep * 0.8))
            else:
                action = ctrl.select_actions(obs, explore=(ep < n_ep * 0.8))

            nobs, reward, done, _, info = env.step(action)

            if isinstance(ctrl, HMADRLFramework):
                prev = local_rewards.copy()
                local_rewards = ctrl.compute_local_rewards(info)
                ctrl.store_transitions(obs, action, local_rewards, reward, nobs, done, prev)
            elif isinstance(ctrl, FlatMADRL):
                ctrl.store_transitions(obs, action, np.full(4, reward / 4), nobs, done)
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
        "method":    method,
        "label":     label,
        "rewards":   rewards,
        "costs":     costs,
        "lolps":     lolps_ep,
        "soc_last":  soc_trajs[-1],
        "rcir":      rcir(costs[-20:]),
        "avg_cost":  float(np.mean(costs[-20:])),
        "avg_lolp":  float(np.mean(lolps_ep[-20:])),
        "avg_reward": float(np.mean(rewards[-20:])),
    }


# ---------------------------------------------------------------------------
def make_plots(results):
    colors = {"hma": "#E84855", "flat": "#FF9F1C", "sa": "#2EC4B6"}

    # --- Convergence ---
    fig, ax = plt.subplots(figsize=(9, 4))
    for r in results:
        m   = r["method"]
        rw  = np.array(r["rewards"])
        sm  = np.convolve(rw, np.ones(5) / 5, mode="valid")
        ax.plot(sm, color=colors[m], lw=2, label=r["label"])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episodic Reward")
    ax.set_title("Training Convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "convergence.png", dpi=150)
    plt.close()

    # --- Cost bar ---
    fig, ax = plt.subplots(figsize=(6, 4))
    base = max(r["avg_cost"] for r in results) + 1e-9
    for r in results:
        sav = max(0, 100 * (1 - r["avg_cost"] / base))
        ax.bar(r["label"], sav, color=colors[r["method"]], alpha=0.85)
    ax.set_ylabel("Cost Reduction (%)")
    ax.set_title("Average Cost Reduction")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "cost_reduction.png", dpi=150)
    plt.close()

    # --- SOC trajectories ---
    fig, ax = plt.subplots(figsize=(9, 4))
    for r in results:
        ax.plot(r["soc_last"], color=colors[r["method"]], lw=2, label=r["label"])
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("SOC (p.u.)")
    ax.set_title("SOC Trajectories – Last Episode")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "soc_trajectories.png", dpi=150)
    plt.close()

    # --- Box plots (rewards) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [r["rewards"][-20:] for r in results]
    lbls = [r["label"] for r in results]
    bp = ax.boxplot(data, labels=lbls, patch_artist=True,
                    medianprops={"color": "black", "lw": 2})
    for patch, r in zip(bp["boxes"], results):
        patch.set_facecolor(colors[r["method"]])
        patch.set_alpha(0.7)
    ax.set_ylabel("Reward (last 20 episodes)")
    ax.set_title("Reward Variability")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "reward_variability.png", dpi=150)
    plt.close()

    print(f"Plots saved → {OUT_DIR}/")


# ---------------------------------------------------------------------------
def print_table(results):
    print()
    print("=" * 72)
    print(f"{'Method':<28} {'Avg Reward':>12} {'Avg Cost':>10} {'RCIR':>8} {'LOLP%':>8}")
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
    print("\nAll done. Check demo_plots/ for figures.")
