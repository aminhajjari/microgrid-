"""
train.py
========
Training loop for SA-DRL, FMA-DRL, and HMA-DRL.

Usage:
    python train.py                    # trains all three, saves checkpoints + plots
    python train.py --method hma       # train only HMA-DRL
    python train.py --episodes 500     # quick smoke-test

Reproduces the convergence curves and comparative metrics in Section 3.3.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from microgrid_env import MicrogridEnv
from hma_drl import HMADRLFramework, FlatMADRL, SingleAgentDRL


# ---------------------------------------------------------------------------
def train_episode(env, controller, batch_size: int = 256, explore: bool = True) -> dict:
    """Run one 24-h episode and return metrics."""
    obs, _ = env.reset()
    total_reward  = 0.0
    total_cost    = 0.0
    local_rewards = np.zeros(4)
    losses        = {}
    soc_traj      = []
    load_losses   = []

    for _ in range(env.T):
        # --- Select actions ---
        if isinstance(controller, HMADRLFramework):
            action = controller.select_actions(obs, local_rewards, explore=explore)
        elif isinstance(controller, FlatMADRL):
            action = controller.select_actions(obs, explore=explore)
        else:   # SA-DRL
            action = controller.select_actions(obs, explore=explore)

        next_obs, reward, done, _, info = env.step(action)

        # --- Compute per-agent rewards ---
        if isinstance(controller, HMADRLFramework):
            prev_local = local_rewards.copy()
            local_rewards = controller.compute_local_rewards(info)
            controller.store_transitions(
                obs, action, local_rewards, reward, next_obs, done, prev_local
            )
        elif isinstance(controller, FlatMADRL):
            local_rewards = np.full(4, reward / 4)   # split evenly
            controller.store_transitions(
                obs, action, local_rewards, next_obs, done
            )
        else:
            controller.store_transitions(obs, action, reward, next_obs, done)

        losses = controller.update_all(batch_size)

        total_reward += reward
        total_cost   += info.get("tariff", 0) * max(0, info.get("p_grid", 0))
        soc_traj.append(info.get("soc_bess", 0.5))
        load_losses.append(info.get("load_loss", 0.0))
        obs = next_obs

    lolp = sum(1 for ll in load_losses if ll > 1.0) / len(load_losses)

    return {
        "reward":      total_reward,
        "cost":        total_cost,
        "soc_traj":    soc_traj,
        "lolp":        lolp,
        "losses":      losses,
    }


# ---------------------------------------------------------------------------
def compute_rcir(costs: list) -> float:
    """Eq. 20: RCIR = 1 - σ(C) / μ(C)"""
    c = np.array(costs)
    mu = c.mean()
    if mu == 0:
        return 1.0
    return float(1 - c.std() / mu)


# ---------------------------------------------------------------------------
def run_training(
    method:      str  = "hma",
    n_episodes:  int  = 500,
    batch_size:  int  = 256,
    device:      str  = "cpu",
    scenario:    str  = "normal",
    save_dir:    Path = Path("checkpoints"),
    verbose:     bool = True,
) -> dict:
    save_dir.mkdir(exist_ok=True)

    env = MicrogridEnv(episode_seed=0, domain_randomize=True, scenario=scenario)

    if method == "hma":
        controller = HMADRLFramework(device=device)
        label = "Hierarchical MA-DRL (Proposed)"
    elif method == "flat":
        controller = FlatMADRL(device=device)
        label = "Flat MA-DRL"
    else:
        controller = SingleAgentDRL(device=device)
        label = "Single-Agent DRL"

    rewards   = []
    costs     = []
    lolps     = []
    soc_final = []

    # Warm-up: fill replay buffer with random transitions
    print(f"[{label}] Warming up replay buffer …")
    obs, _ = env.reset()
    for _ in range(min(1000, n_episodes * env.T // 5)):
        action = env.action_space.sample()
        nobs, rew, done, _, info = env.step(action)
        if isinstance(controller, HMADRLFramework):
            lr = controller.compute_local_rewards(info)
            controller.store_transitions(obs, action, lr, rew, nobs, done, np.zeros(4))
        elif isinstance(controller, FlatMADRL):
            controller.store_transitions(obs, action, np.full(4, rew / 4), nobs, done)
        else:
            controller.store_transitions(obs, action, rew, nobs, done)
        obs = nobs if not done else env.reset()[0]

    # Training
    t0 = time.time()
    for ep in range(1, n_episodes + 1):
        env.episode_seed = ep
        result = train_episode(env, controller, batch_size, explore=True)
        rewards.append(result["reward"])
        costs.append(result["cost"])
        lolps.append(result["lolp"])
        soc_final.append(result["soc_traj"])

        if verbose and ep % max(1, n_episodes // 10) == 0:
            elapsed = time.time() - t0
            avg_r = np.mean(rewards[-50:])
            print(
                f"  Ep {ep:5d}/{n_episodes} | "
                f"Avg reward(50) = {avg_r:7.2f} | "
                f"Cost = {result['cost']:.3f} | "
                f"LOLP = {result['lolp']:.3f} | "
                f"Elapsed = {elapsed:.1f}s"
            )

    # Evaluation (greedy)
    print(f"\n[{label}] Evaluating (10 greedy runs) …")
    eval_costs   = []
    eval_rewards = []
    eval_lolps   = []
    for run in range(10):
        env.episode_seed = 1000 + run
        result = train_episode(env, controller, batch_size=1, explore=False)
        eval_costs.append(result["cost"])
        eval_rewards.append(result["reward"])
        eval_lolps.append(result["lolp"])

    rcir = compute_rcir(eval_costs)

    summary = {
        "method":        method,
        "label":         label,
        "rewards":       rewards,
        "costs":         costs,
        "lolps":         lolps,
        "soc_traj_last": soc_final[-1] if soc_final else [],
        "eval_costs":    eval_costs,
        "eval_rewards":  eval_rewards,
        "eval_lolps":    eval_lolps,
        "rcir":          rcir,
        "avg_cost_last50": float(np.mean(costs[-50:])),
        "avg_reward_last50": float(np.mean(rewards[-50:])),
        "avg_lolp":      float(np.mean(eval_lolps)),
    }

    # Save checkpoint
    ckpt_path = save_dir / f"{method}_results.npz"
    np.savez(str(ckpt_path), **{k: v for k, v in summary.items() if isinstance(v, (list, float, str))})
    print(f"  Saved → {ckpt_path}")

    return summary


# ---------------------------------------------------------------------------
def plot_all_results(results: list, out_dir: Path):
    """
    Reproduce key figures from the paper:
      Fig 3 – Training convergence
      Fig 4 – Cost savings bar chart
      Fig 6 – SOC trajectories (24 h)
      Fig 12 – Statistical variability (box plots)
    """
    out_dir.mkdir(exist_ok=True)
    colors = {"hma": "#E84855", "flat": "#FF9F1C", "sa": "#2EC4B6"}
    labels = {r["method"]: r["label"] for r in results}

    # ---- Figure 3: Training Convergence ----
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        m   = r["method"]
        rws = r["rewards"]
        ep  = np.arange(1, len(rws) + 1)
        # smooth
        smooth = np.convolve(rws, np.ones(20) / 20, mode="valid")
        ax.plot(ep[:len(smooth)], smooth, color=colors[m], lw=2, label=labels[m])
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("Average Episodic Reward")
    ax.set_title("Training Convergence of DRL Architectures")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "fig3_training_convergence.png", dpi=150)
    plt.close(fig)

    # ---- Figure 4: Cost Savings Bar Chart ----
    avg_costs = {r["method"]: r["avg_cost_last50"] for r in results}
    baseline  = max(avg_costs.values()) + 1e-9
    savings   = {m: 100 * (1 - c / baseline) for m, c in avg_costs.items()}
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, s in savings.items():
        ax.bar(labels[m], max(s, 0), color=colors[m], alpha=0.85)
    ax.set_ylabel("Cost Reduction (%)")
    ax.set_title("Average Cost Reduction vs. Baseline")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "fig4_cost_reduction.png", dpi=150)
    plt.close(fig)

    # ---- Figure 6: SOC Trajectories ----
    fig, ax = plt.subplots(figsize=(9, 4))
    for r in results:
        traj = r["soc_traj_last"]
        if traj:
            ax.plot(traj, color=colors[r["method"]], lw=2, label=labels[r["method"]])
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("SOC (p.u.)")
    ax.set_title("SOC Trajectories over 24 h Horizon")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "fig6_soc_trajectories.png", dpi=150)
    plt.close(fig)

    # ---- Figure 12: Statistical Variability ----
    fig, ax = plt.subplots(figsize=(7, 5))
    data   = [r["eval_rewards"] for r in results]
    lbls   = [labels[r["method"]] for r in results]
    bp = ax.boxplot(data, labels=lbls, patch_artist=True,
                    medianprops={"color": "black", "lw": 2})
    for patch, r in zip(bp["boxes"], results):
        patch.set_facecolor(colors[r["method"]])
        patch.set_alpha(0.7)
    ax.set_ylabel("Evaluation Reward")
    ax.set_title("Statistical Variability of Reward (10 runs)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "fig12_statistical_variability.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to: {out_dir}")


# ---------------------------------------------------------------------------
def print_comparison_table(results: list):
    """Reproduce Table 4 in the paper."""
    print("\n" + "=" * 72)
    print(f"{'Method':<28} {'Avg Cost':>10} {'RCIR':>8} {'LOLP(%)':>9} {'Episodes':>10}")
    print("=" * 72)
    for r in results:
        episodes_to_converge = _convergence_episode(r["rewards"])
        print(
            f"{r['label']:<28} "
            f"{r['avg_cost_last50']:>10.4f} "
            f"{r['rcir']:>8.3f} "
            f"{100*r['avg_lolp']:>9.3f} "
            f"{episodes_to_converge:>10d}"
        )
    print("=" * 72)


def _convergence_episode(rewards: list, pct: float = 0.95) -> int:
    """Episode at which reward first reaches pct of final value."""
    final = np.mean(rewards[-20:]) if len(rewards) >= 20 else rewards[-1]
    threshold = pct * final
    for i, r in enumerate(rewards):
        if r >= threshold:
            return i + 1
    return len(rewards)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",   default="all", choices=["all", "hma", "flat", "sa"])
    parser.add_argument("--episodes", type=int, default=300,
                        help="Episodes per method (paper uses 5000; 300 for quick test)")
    parser.add_argument("--batch",    type=int, default=256)
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--scenario", default="normal",
                        choices=["normal", "crit_load", "pv_outage", "dynamic_price", "high_res"])
    args = parser.parse_args()

    methods = ["sa", "flat", "hma"] if args.method == "all" else [args.method]
    results = []

    for m in methods:
        print(f"\n{'='*60}")
        print(f"  Training  {m.upper()}  ({args.episodes} episodes, scenario={args.scenario})")
        print(f"{'='*60}")
        res = run_training(
            method=m,
            n_episodes=args.episodes,
            batch_size=args.batch,
            device=args.device,
            scenario=args.scenario,
        )
        results.append(res)

    if len(results) > 1:
        print_comparison_table(results)

    out_dir = Path("plots")
    plot_all_results(results, out_dir)
    print("\nDone.")