"""
train.py  (v5 — ARW integrated)
========
Training loop for SA-DRL, FMA-DRL, and HMA-DRL.

==========================================================================
CHANGES vs. v4  (every change is marked ARW-*)
==========================================================================

ARW-1  train_episode() now calls get_reward_weights(obs) before env.step()
       for HMA, passing adaptive weights to the environment.
       SA and Flat call env.step(action) with no weights — identical to v4.

ARW-2  record_reward() called after each env.step() for HMA, feeding the
       scalar reward into the REINFORCE buffer of the AdaptiveRewardWeighter.

ARW-3  update_arw() called once per episode end for HMA, performing one
       REINFORCE gradient step on the AdaptiveRewardWeightNetwork.

ARW-4  load_weights() now loads ARW weights from checkpoint when present
       (backwards compatible — old checkpoints without 'arw' key still load).

ARW-5  summary dict includes avg_arw_loss for logging/plotting.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from microgrid_env import MicrogridEnv
from hma_drl import HMADRLFramework, FlatMADRL, SingleAgentDRL, _softmax, _save_hma_weights


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WARMUP_STEPS         = 2000
DIVERGENCE_THRESHOLD = -20.0
REPORT_WINDOW        = 50


# ---------------------------------------------------------------------------
# Warm-up helper
# ---------------------------------------------------------------------------
def warmup_buffer(env, controller, steps: int = WARMUP_STEPS):
    """Fill replay buffers with random transitions before training starts."""
    obs, _ = env.reset()
    for _ in range(steps):
        action = env.action_space.sample()

        if isinstance(controller, HMADRLFramework):
            # ARW-1: use fixed paper weights during warm-up (arw is in warm-up too)
            rw   = controller.get_reward_weights(obs)
            nobs, rew, done, _, info = env.step(action, reward_weights=rw)
            controller.record_reward(rew)
            lr    = controller.compute_local_rewards(info)
            omega = _softmax(np.zeros(4))
            controller.store_transitions(
                obs, action, lr, rew, nobs, done, np.zeros(4), omega
            )
        elif isinstance(controller, FlatMADRL):
            nobs, rew, done, _, info = env.step(action)
            controller.store_transitions(
                obs, action, np.full(4, rew / 4), nobs, done
            )
        else:
            nobs, rew, done, _, info = env.step(action)
            controller.store_transitions(obs, action, rew, nobs, done)

        obs = nobs if not done else env.reset()[0]


# ---------------------------------------------------------------------------
# Single episode  (ARW-1, ARW-2, ARW-3)
# ---------------------------------------------------------------------------
def train_episode(env, controller, batch_size: int = 256,
                  explore: bool = True) -> dict:
    obs, _        = env.reset()
    total_reward  = 0.0
    total_cost    = 0.0
    local_rewards = np.zeros(4)
    soc_traj      = []
    load_losses   = []

    for _ in range(env.T):
        if isinstance(controller, HMADRLFramework):
            # ARW-1: adaptive weights for this timestep
            rw = controller.get_reward_weights(obs)
            action, omega = controller.select_actions(obs, local_rewards, explore=explore)
            next_obs, reward, done, _, info = env.step(action, reward_weights=rw)
            # ARW-2: feed reward into REINFORCE buffer
            controller.record_reward(reward)
        else:
            action = controller.select_actions(obs, explore=explore)
            # SA / Flat: no adaptive weights — identical to v4
            next_obs, reward, done, _, info = env.step(action)

        if isinstance(controller, HMADRLFramework):
            prev_local    = local_rewards.copy()
            local_rewards = controller.compute_local_rewards(info)
            controller.store_transitions(
                obs, action, local_rewards, reward, next_obs, done, prev_local, omega
            )
        elif isinstance(controller, FlatMADRL):
            local_rewards = np.full(4, reward / 4)
            controller.store_transitions(obs, action, local_rewards, next_obs, done)
        else:
            controller.store_transitions(obs, action, reward, next_obs, done)

        controller.update_all(batch_size)

        total_reward += reward
        total_cost   += info.get("tariff", 0) * max(0, info.get("p_grid", 0))
        soc_traj.append(info.get("soc_bess", 0.5))
        load_losses.append(info.get("load_loss", 0.0))
        obs = next_obs

    # ARW-3: episode-end REINFORCE update for ARW (no-op for SA/Flat)
    arw_info = {}
    if isinstance(controller, HMADRLFramework):
        arw_info = controller.update_arw()

    lolp = sum(1 for ll in load_losses if ll > 1.0) / len(load_losses)
    return {
        "reward":   total_reward,
        "cost":     total_cost,
        "soc_traj": soc_traj,
        "lolp":     lolp,
        "arw_loss": arw_info.get("arw_loss", 0.0),
    }


# ---------------------------------------------------------------------------
# RCIR
# ---------------------------------------------------------------------------
def compute_rcir(costs: list) -> float:
    c  = np.array(costs, dtype=float)
    mu = c.mean()
    if mu == 0:
        return 1.0
    return float(np.clip(1 - c.std() / mu, 0, 1))


# ---------------------------------------------------------------------------
# Weight I/O helpers  (ARW-4)
# ---------------------------------------------------------------------------
def save_weights(controller, save_dir: Path, method: str):
    """Save weights for SA and Flat (HMA uses save_if_best instead)."""
    w = {}
    if isinstance(controller, FlatMADRL):
        for name, agent in controller.agents.items():
            w[f"{name}_actor"]  = agent.actor.state_dict()
            w[f"{name}_critic"] = agent.critic.state_dict()
    else:  # SingleAgentDRL
        w["actor"]  = controller.agent.actor.state_dict()
        w["critic"] = controller.agent.critic.state_dict()
    torch.save(w, save_dir / f"{method}_weights.pt")
    print(f"  Weights saved → {save_dir}/{method}_weights.pt")


def load_weights(controller, save_dir: Path, method: str):
    path = save_dir / f"{method}_weights.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"No weights at {path} — run normal scenario training first."
        )
    w = torch.load(path, map_location="cpu")
    if isinstance(controller, HMADRLFramework):
        for name, agent in controller.agents.items():
            agent.actor.load_state_dict(w[f"{name}_actor"])
            agent.critic.load_state_dict(w[f"{name}_critic"])
        controller.supervisor.actor.load_state_dict(w["supervisor_actor"])
        controller.supervisor.critic.load_state_dict(w["supervisor_critic"])
        # ARW-4: load ARW weights if present (backwards compatible)
        if "arw" in w:
            controller.arw.load_state_dict(w["arw"])
    elif isinstance(controller, FlatMADRL):
        for name, agent in controller.agents.items():
            agent.actor.load_state_dict(w[f"{name}_actor"])
            agent.critic.load_state_dict(w[f"{name}_critic"])
    else:
        controller.agent.actor.load_state_dict(w["actor"])
        controller.agent.critic.load_state_dict(w["critic"])
    print(f"  Weights loaded ← {path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def run_training(
    method:     str  = "hma",
    n_episodes: int  = 5000,
    batch_size: int  = 256,
    device:     str  = "cpu",
    scenario:   str  = "normal",
    save_dir:   Path = Path("/home/gkianfar/scratch/Amin/MSH/output/checkpoints"),
    verbose:    bool = True,
    eval_only:  bool = False,
) -> dict:

    save_dir.mkdir(exist_ok=True)

    env = MicrogridEnv(episode_seed=0, domain_randomize=True, scenario=scenario)

    if method == "hma":
        controller = HMADRLFramework(device=device)
        label      = "Hierarchical MA-DRL (Proposed)"
    elif method == "flat":
        controller = FlatMADRL(device=device)
        label      = "Flat MA-DRL"
    else:
        controller = SingleAgentDRL(device=device)
        label      = "Single-Agent DRL"

    if eval_only:
        print(f"[{label}] EVAL-ONLY mode — loading trained weights …")
        load_weights(controller, save_dir, method)
        n_episodes = 0

    if not eval_only:
        print(f"[{label}] Warming up replay buffer ({WARMUP_STEPS} steps) …")
        warmup_buffer(env, controller, WARMUP_STEPS)

    rewards, costs, lolps, soc_final, arw_losses = [], [], [], [], []
    t0 = time.time()
    divergence_strikes = 0

    for ep in range(1, n_episodes + 1):
        env.episode_seed = ep
        result = train_episode(env, controller, batch_size, explore=True)
        rewards.append(result["reward"])
        costs.append(result["cost"])
        lolps.append(result["lolp"])
        soc_final.append(result["soc_traj"])
        arw_losses.append(result["arw_loss"])   # ARW-5

        if verbose and ep % max(1, n_episodes // 10) == 0:
            elapsed  = time.time() - t0
            avg_r    = np.mean(rewards[-REPORT_WINDOW:])
            avg_arw  = np.mean(arw_losses[-REPORT_WINDOW:])
            arw_str  = f" | ARW_loss={avg_arw:.4f}" if method == "hma" else ""
            print(
                f"  Ep {ep:5d}/{n_episodes} | "
                f"Avg reward({REPORT_WINDOW}) = {avg_r:7.2f} | "
                f"Cost = {result['cost']:.3f} | "
                f"LOLP = {result['lolp']:.3f} | "
                f"Elapsed = {elapsed:.1f}s"
                f"{arw_str}"
            )

            if method == "hma" and avg_r < DIVERGENCE_THRESHOLD:
                divergence_strikes += 1
                if divergence_strikes >= 2:
                    print(
                        f"\n  ⚠️  WARNING: HMA avg reward has been below "
                        f"{DIVERGENCE_THRESHOLD:.1f} for 2 consecutive windows.\n"
                        f"  This seed may be diverging.  Consider restarting.\n"
                    )
            else:
                divergence_strikes = 0

    if not eval_only:
        if method == "hma":
            probe_env = MicrogridEnv(episode_seed=999, domain_randomize=False)
            probe     = train_episode(probe_env, controller, batch_size=1, explore=False)
            saved     = controller.save_if_best(probe["reward"], save_dir, method)
            if not saved:
                print(f"  (not saved — current best is {controller.best_eval_reward:.2f})")
        else:
            save_weights(controller, save_dir, method)

    print(f"\n[{label}] Evaluating (10 greedy runs) …")
    eval_costs, eval_rewards, eval_lolps = [], [], []
    for run in range(10):
        env.episode_seed = 1000 + run
        result = train_episode(env, controller, batch_size=1, explore=False)
        eval_costs.append(result["cost"])
        eval_rewards.append(result["reward"])
        eval_lolps.append(result["lolp"])

    rcir_val = compute_rcir(eval_costs)

    summary = {
        "method":            method,
        "label":             label,
        "scenario":          scenario,
        "rewards":           rewards,
        "costs":             costs,
        "lolps":             lolps,
        "arw_losses":        arw_losses,
        "soc_traj_last":     soc_final[-1] if soc_final else [],
        "eval_costs":        eval_costs,
        "eval_rewards":      eval_rewards,
        "eval_lolps":        eval_lolps,
        "rcir":              rcir_val,
        "avg_cost_last50":   float(np.mean(costs[-50:]))   if costs   else 0.0,
        "avg_reward_last50": float(np.mean(rewards[-50:])) if rewards else 0.0,
        "avg_lolp":          float(np.mean(eval_lolps)),
        "avg_arw_loss":      float(np.mean(arw_losses[-50:])) if arw_losses else 0.0,
    }

    ckpt_path = save_dir / f"{method}_{scenario}_results.npz"
    np.savez(
        str(ckpt_path),
        **{k: v for k, v in summary.items() if isinstance(v, (list, float, str))}
    )
    print(f"  Saved → {ckpt_path}")

    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_all_results(results: list, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    colors = {"hma": "#E84855", "flat": "#FF9F1C", "sa": "#2EC4B6"}
    labels = {r["method"]: r["label"] for r in results}

    # Fig 3 – Training Convergence
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        rws = r["rewards"]
        if len(rws) < 20:
            continue
        smooth = np.convolve(rws, np.ones(20) / 20, mode="valid")
        ax.plot(np.arange(1, len(smooth) + 1), smooth,
                color=colors[r["method"]], lw=2, label=labels[r["method"]])
    ax.set_xlabel("Training Episodes"); ax.set_ylabel("Average Episodic Reward")
    ax.set_title("Training Convergence of DRL Architectures")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(out_dir / "fig3_training_convergence.png", dpi=150); plt.close(fig)

    # Fig 4 – Cost Savings Bar Chart
    avg_costs = {r["method"]: r["avg_cost_last50"] for r in results}
    baseline  = max(avg_costs.values()) + 1e-9
    savings   = {m: 100 * (1 - c / baseline) for m, c in avg_costs.items()}
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, s in savings.items():
        ax.bar(labels[m], max(s, 0), color=colors[m], alpha=0.85)
    ax.set_ylabel("Cost Reduction (%)"); ax.set_title("Average Cost Reduction vs. Baseline")
    ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    fig.savefig(out_dir / "fig4_cost_reduction.png", dpi=150); plt.close(fig)

    # Fig 6 – SOC Trajectories
    fig, ax = plt.subplots(figsize=(9, 4))
    for r in results:
        traj = r["soc_traj_last"]
        if traj:
            ax.plot(traj, color=colors[r["method"]], lw=2, label=labels[r["method"]])
    ax.set_xlabel("Time (h)"); ax.set_ylabel("SOC (p.u.)")
    ax.set_title("SOC Trajectories over 24 h Horizon")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(out_dir / "fig6_soc_trajectories.png", dpi=150); plt.close(fig)

    # Fig 12 – Statistical Variability
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [r["eval_rewards"] for r in results]
    lbls = [labels[r["method"]] for r in results]
    bp   = ax.boxplot(data, tick_labels=lbls, patch_artist=True,
                      medianprops={"color": "black", "lw": 2})
    for patch, r in zip(bp["boxes"], results):
        patch.set_facecolor(colors[r["method"]]); patch.set_alpha(0.7)
    ax.set_ylabel("Evaluation Reward")
    ax.set_title("Statistical Variability of Reward (10 runs)")
    ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    fig.savefig(out_dir / "fig12_statistical_variability.png", dpi=150); plt.close(fig)

    # ARW-5: ARW loss curve (HMA only)
    hma_results = [r for r in results if r["method"] == "hma"]
    if hma_results and hma_results[0].get("arw_losses"):
        arw_l = hma_results[0]["arw_losses"]
        if len(arw_l) >= 20:
            fig, ax = plt.subplots(figsize=(9, 4))
            smooth = np.convolve(arw_l, np.ones(20) / 20, mode="valid")
            ax.plot(smooth, color="#E84855", lw=2)
            ax.set_xlabel("Episode"); ax.set_ylabel("ARW REINFORCE Loss")
            ax.set_title("Adaptive Reward Weighter Training Loss (HMA)")
            ax.grid(alpha=0.3); plt.tight_layout()
            fig.savefig(out_dir / "fig_arw_loss.png", dpi=150); plt.close(fig)

    print(f"\nPlots saved to: {out_dir}")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def print_comparison_table(results: list):
    print("\n" + "=" * 80)
    print(f"{'Method':<28} {'Avg Cost':>10} {'RCIR':>8} {'LOLP(%)':>9} "
          f"{'Episodes':>10} {'ARW Loss':>10}")
    print("=" * 80)
    for r in results:
        ep_conv  = _convergence_episode(r["rewards"])
        arw_loss = r.get("avg_arw_loss", 0.0)
        arw_str  = f"{arw_loss:>10.4f}" if r["method"] == "hma" else f"{'N/A':>10}"
        print(
            f"{r['label']:<28} "
            f"{r['avg_cost_last50']:>10.4f} "
            f"{r['rcir']:>8.3f} "
            f"{100*r['avg_lolp']:>9.3f} "
            f"{ep_conv:>10d} "
            f"{arw_str}"
        )
    print("=" * 80)


def _convergence_episode(rewards: list, pct: float = 0.95) -> int:
    if not rewards:
        return 0
    final     = np.mean(rewards[-20:]) if len(rewards) >= 20 else rewards[-1]
    threshold = pct * final
    for i, r in enumerate(rewards):
        if r >= threshold:
            return i + 1
    return len(rewards)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",    default="all",
                        choices=["all", "hma", "flat", "sa"])
    parser.add_argument("--episodes",  type=int, default=5000)
    parser.add_argument("--batch",     type=int, default=256)
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--scenario",  default="normal",
                        choices=["normal", "crit_load", "pv_outage",
                                 "dynamic_price", "high_res"])
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    methods = ["sa", "flat", "hma"] if args.method == "all" else [args.method]
    results = []

    for m in methods:
        print(f"\n{'='*60}")
        print(f"  Training  {m.upper()}  ({args.episodes} episodes, "
              f"scenario={args.scenario})")
        print(f"{'='*60}")
        res = run_training(
            method     = m,
            n_episodes = args.episodes,
            batch_size = args.batch,
            device     = args.device,
            scenario   = args.scenario,
            save_dir   = Path("/home/gkianfar/scratch/Amin/MSH/output/checkpoints"),
            eval_only  = args.eval_only,
        )
        results.append(res)

    if len(results) > 1:
        print_comparison_table(results)

    out_dir = Path("/home/gkianfar/scratch/Amin/MSH/output/plots")
    plot_all_results(results, out_dir)
    print("\nDone.")
