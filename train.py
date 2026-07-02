"""
train.py  — v2 (fixed)

FIXES vs v1:
  FIX-1  train_episode() now records and returns rur, degradation,
         exec_time so all paper Table 4 columns are populated.
  FIX-2  run_training() default episodes raised to 10,000 to match
         the paper specification (Section 2.5).
  FIX-3  print_comparison_table() outputs all 8 paper Table 4 columns
         and shows paper targets for direct comparison.
  FIX-4  warmup_buffer() and train_episode() unchanged except for new
         metric collection — ARW integration identical to v1.
"""

import argparse
import time
import time as _time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from microgrid_env import MicrogridEnv
from hma_drl import HMADRLFramework, FlatMADRL, SingleAgentDRL, _softmax, _save_hma_weights
from metrics_v2 import battery_degradation


WARMUP_STEPS         = 2000
DIVERGENCE_THRESHOLD = -20.0
REPORT_WINDOW        = 50


# ---------------------------------------------------------------------------
def warmup_buffer(env, controller, steps: int = WARMUP_STEPS):
    obs, _ = env.reset()
    for _ in range(steps):
        action = env.action_space.sample()
        if isinstance(controller, HMADRLFramework):
            rw   = controller.get_reward_weights(obs)
            nobs, rew, done, _, info = env.step(action, reward_weights=rw)
            controller.record_reward(controller.compute_arw_reward(info))  # v3: fixed KPI
            lr    = controller.compute_local_rewards(info, rw)             # v3: weight-aware
            omega = np.zeros(2)   # supervisor acts only on bess+ev (N_MOD=2), not 4
            controller.store_transitions(obs, action, lr, rew, nobs, done, np.zeros(4), omega)
        elif isinstance(controller, FlatMADRL):
            nobs, rew, done, _, info = env.step(action)
            controller.store_transitions(obs, action, np.full(4, rew / 4), nobs, done)
        else:
            nobs, rew, done, _, info = env.step(action)
            controller.store_transitions(obs, action, rew, nobs, done)
        obs = nobs if not done else env.reset()[0]


# ---------------------------------------------------------------------------
def train_episode(env, controller, batch_size: int = 256, explore: bool = True) -> dict:
    obs, _        = env.reset()
    total_reward  = 0.0
    total_cost    = 0.0
    local_rewards = np.zeros(4)
    soc_traj      = []
    load_losses   = []
    p_pv_list     = []      # FIX-1
    p_load_list   = []      # FIX-1
    t_ep_start    = _time.time()   # FIX-1

    for _ in range(env.T):
        if isinstance(controller, HMADRLFramework):
            rw = controller.get_reward_weights(obs)
            action, omega = controller.select_actions(obs, local_rewards, explore=explore)
            next_obs, reward, done, _, info = env.step(action, reward_weights=rw)
            controller.record_reward(controller.compute_arw_reward(info))  # v3: fixed KPI
        else:
            action = controller.select_actions(obs, explore=explore)
            next_obs, reward, done, _, info = env.step(action)

        if isinstance(controller, HMADRLFramework):
            prev_local    = local_rewards.copy()
            local_rewards = controller.compute_local_rewards(info, rw)  # v3: weight-aware
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
        p_pv_list.append(info.get("p_pv", 0.0))      # FIX-1
        p_load_list.append(info.get("p_load", 30.0))  # FIX-1
        obs = next_obs

    arw_info = {}
    if isinstance(controller, HMADRLFramework):
        arw_info = controller.update_arw()

    lolp = sum(1 for ll in load_losses if ll > 1.0) / max(len(load_losses), 1)

    # FIX-1: compute RUR (Eq. 22) and degradation (Eq. 19)
    rur = sum(p_pv_list) / max(sum(p_load_list), 1e-9)
    deg = battery_degradation(soc_traj)
    exec_time = _time.time() - t_ep_start

    return {
        "reward":      total_reward,
        "cost":        total_cost,
        "soc_traj":    soc_traj,
        "lolp":        lolp,
        "arw_loss":    arw_info.get("arw_loss", 0.0),
        "rur":         rur,        # FIX-1
        "degradation": deg,        # FIX-1
        "exec_time":   exec_time,  # FIX-1
    }


# ---------------------------------------------------------------------------
def compute_rcir(costs: list) -> float:
    c  = np.array(costs, dtype=float)
    mu = c.mean()
    if mu == 0:
        return 1.0
    return float(np.clip(1 - c.std() / mu, 0, 1))


# ---------------------------------------------------------------------------
def save_weights(controller, save_dir: Path, method: str):
    if isinstance(controller, HMADRLFramework):      # covers hma_fixed too
        _save_hma_weights(controller, save_dir, method)
        return
    w = {}
    if isinstance(controller, FlatMADRL):
        for name, agent in controller.agents.items():
            w[f"{name}_actor"]  = agent.actor.state_dict()
            w[f"{name}_critic"] = agent.critic.state_dict()
    else:
        w["actor"]  = controller.agent.actor.state_dict()
        w["critic"] = controller.agent.critic.state_dict()
    torch.save(w, save_dir / f"{method}_weights.pt")
    print(f"  Weights saved → {save_dir}/{method}_weights.pt")


def load_weights(controller, save_dir: Path, method: str):
    path = save_dir / f"{method}_weights.pt"
    if not path.exists():
        raise FileNotFoundError(f"No weights at {path}")
    w = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(controller, HMADRLFramework):
        for name, agent in controller.agents.items():
            agent.actor.load_state_dict(w[f"{name}_actor"])
            agent.critic.load_state_dict(w[f"{name}_critic"])
        controller.supervisor.actor.load_state_dict(w["supervisor_actor"])
        controller.supervisor.critic.load_state_dict(w["supervisor_critic"])
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
def run_training(
    method:     str  = "hma",
    n_episodes: int  = 10000,   # FIX-2: raised from 5000 to 10000
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
        controller.set_scenario(scenario) 
    elif method == "flat":
        controller = FlatMADRL(device=device)
        label      = "Flat MA-DRL"

    elif method == "hma_fixed":
        from hma_drl import HMADRLFixedWeights
        controller = HMADRLFixedWeights(device=device)
        label      = "HMA + Fixed Weights"
  
    else:
        controller = SingleAgentDRL(device=device)
        label      = "Single-Agent DRL"

    if eval_only:
        print(f"[{label}] EVAL-ONLY — loading weights …")
        load_weights(controller, save_dir, method)
        n_episodes = 0

    if not eval_only:
        print(f"[{label}] Warming up replay buffer ({WARMUP_STEPS} steps) …")
        warmup_buffer(env, controller, WARMUP_STEPS)

    rewards, costs, lolps       = [], [], []
    rurs, degradations, exec_times = [], [], []   # FIX-1
    soc_final, arw_losses       = [], []
    t0                          = time.time()
    divergence_strikes          = 0

    for ep in range(1, n_episodes + 1):
        env.episode_seed = ep
        result = train_episode(env, controller, batch_size, explore=True)
        rewards.append(result["reward"])
        costs.append(result["cost"])
        lolps.append(result["lolp"])
        rurs.append(result["rur"])               # FIX-1
        degradations.append(result["degradation"]) # FIX-1
        exec_times.append(result["exec_time"])   # FIX-1
        soc_final.append(result["soc_traj"])
        arw_losses.append(result["arw_loss"])

        if verbose and ep % max(1, n_episodes // 10) == 0:
            elapsed = time.time() - t0
            avg_r   = np.mean(rewards[-REPORT_WINDOW:])
            avg_arw = np.mean(arw_losses[-REPORT_WINDOW:])
            arw_str = f" | ARW_loss={avg_arw:.4f}" if method == "hma" else ""
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
                    print(f"\n  ⚠️  HMA avg reward < {DIVERGENCE_THRESHOLD:.1f} for 2 windows. "
                          f"Consider restarting.\n")
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
        "method":             method,
        "label":              label,
        "scenario":           scenario,
        "rewards":            rewards,
        "costs":              costs,
        "lolps":              lolps,
        "rurs":               rurs,
        "degradations":       degradations,
        "exec_times":         exec_times,
        "arw_losses":         arw_losses,
        "soc_traj_last":      soc_final[-1] if soc_final else [],
        "eval_costs":         eval_costs,
        "eval_rewards":       eval_rewards,
        "eval_lolps":         eval_lolps,
        "rcir":               rcir_val,
        "avg_cost_last50":    float(np.mean(costs[-50:]))        if costs        else 0.0,
        "avg_reward_last50":  float(np.mean(rewards[-50:]))      if rewards      else 0.0,
        "avg_lolp":           float(np.mean(eval_lolps)),
        "avg_rur_last50":     float(np.mean(rurs[-50:]))         if rurs         else 0.0,
        "avg_deg_last50":     float(np.mean(degradations[-50:])) if degradations else 0.0,
        "avg_exec_time":      float(np.mean(exec_times[-50:]))   if exec_times   else 0.0,
        "avg_arw_loss":       float(np.mean(arw_losses[-50:]))   if arw_losses   else 0.0,
    }

    ckpt_path = save_dir / f"{method}_{scenario}_results.npz"
    np.savez(str(ckpt_path),
             **{k: v for k, v in summary.items() if isinstance(v, (list, float, str))})
    print(f"  Saved → {ckpt_path}")
    return summary


# ---------------------------------------------------------------------------
def plot_all_results(results: list, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    colors = {"hma": "#E84855", "flat": "#FF9F1C", "sa": "#2EC4B6"}
    labels = {r["method"]: r["label"] for r in results}

    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        rws = r["rewards"]
        if len(rws) < 20:
            continue
        smooth = np.convolve(rws, np.ones(20) / 20, mode="valid")
        ax.plot(np.arange(1, len(smooth)+1), smooth,
                color=colors[r["method"]], lw=2, label=labels[r["method"]])
    ax.set_xlabel("Training Episodes"); ax.set_ylabel("Average Episodic Reward")
    ax.set_title("Training Convergence of DRL Architectures")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(out_dir / "fig3_training_convergence.png", dpi=150); plt.close(fig)

    avg_costs = {r["method"]: r["avg_cost_last50"] for r in results}
    baseline  = max(avg_costs.values()) + 1e-9
    savings   = {m: 100*(1 - c/baseline) for m, c in avg_costs.items()}
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, s in savings.items():
        ax.bar(labels[m], max(s, 0), color=colors[m], alpha=0.85)
    ax.set_ylabel("Cost Reduction (%)"); ax.set_title("Average Cost Reduction vs. Baseline")
    ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    fig.savefig(out_dir / "fig4_cost_reduction.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for r in results:
        if r["soc_traj_last"]:
            ax.plot(r["soc_traj_last"], color=colors[r["method"]], lw=2, label=labels[r["method"]])
    ax.set_xlabel("Time (h)"); ax.set_ylabel("SOC (p.u.)")
    ax.set_title("SOC Trajectories over 24h Horizon")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(out_dir / "fig6_soc_trajectories.png", dpi=150); plt.close(fig)

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

    hma_r = [r for r in results if r["method"] == "hma"]
    if hma_r and hma_r[0].get("arw_losses"):
        arw_l = hma_r[0]["arw_losses"]
        if len(arw_l) >= 20:
            fig, ax = plt.subplots(figsize=(9, 4))
            smooth = np.convolve(arw_l, np.ones(20)/20, mode="valid")
            ax.plot(smooth, color="#E84855", lw=2)
            ax.set_xlabel("Episode"); ax.set_ylabel("ARW REINFORCE Loss")
            ax.set_title("Adaptive Reward Weighter Training Loss (HMA)")
            ax.grid(alpha=0.3); plt.tight_layout()
            fig.savefig(out_dir / "fig_arw_loss.png", dpi=150); plt.close(fig)

    print(f"\nPlots saved to: {out_dir}")



# _____________________________________________________________________
def compute_rule_based_baseline(n_episodes: int = 50) -> float:
    """
    Fixed-dispatch baseline: no storage, no flexibility actions.
    All load served from grid only. Used as cost denominator for
    cost reduction % — same reference as the paper's Table 4.
    """
    env = MicrogridEnv(episode_seed=0, domain_randomize=False)
    costs = []
    for ep in range(n_episodes):
        env.episode_seed = ep
        obs, _ = env.reset()
        ep_cost = 0.0
        for _ in range(env.T):
            action = np.zeros(4)  # no BESS, no EV, no flex shift, no export
            _, _, done, _, info = env.step(action)
            ep_cost += info["tariff"] * max(0, info["p_grid"])
        costs.append(ep_cost)
    baseline = float(np.mean(costs))
    print(f"  Rule-based baseline cost: {baseline:.4f}")
    return baseline
# ---------------------------------------------------------------------------
# FIX-3: updated comparison table with all 8 paper Table 4 columns
# ---------------------------------------------------------------------------
def print_comparison_table(results: list, rule_based_baseline: float = None):
    # ── cost baseline ──────────────────────────────────────────────────────
    if rule_based_baseline is not None and rule_based_baseline > 0:
        baseline_cost = rule_based_baseline
    else:
        # fallback if baseline wasn't computed
        baseline_cost = max(
            (r["avg_cost_last50"] for r in results), default=1.0
        ) + 1e-9

    # ── degradation baseline (SA-DRL = 1.0 reference) ─────────────────────
    sa_deg = next(
        (r["avg_deg_last50"] for r in results if r["method"] == "sa"), 1.0
    )
    if sa_deg <= 0:
        sa_deg = 1.0

    # ── header ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 118)
    print(
        f"{'Method':<30} {'CostRed%':>8} {'RUR':>6} {'RCIR':>6} "
        f"{'Conv.Ep':>8} {'Deg.Idx':>8} {'s/ep':>6} {'LOLP%':>7}"
    )
    print("=" * 118)

    # ── your results ───────────────────────────────────────────────────────
    for r in results:
        cost_red = max(0.0, 100.0 * (1.0 - r["avg_cost_last50"] / baseline_cost))
        deg_rel  = r["avg_deg_last50"] / sa_deg
        ep_conv  = _convergence_episode(r.get("rewards", []))
        tag      = " [+ARW]" if r["method"] == "hma" else ""
        print(
            f"{r['label'] + tag:<30} "
            f"{cost_red:>8.1f} "
            f"{r['avg_rur_last50']:>6.3f} "
            f"{r['rcir']:>6.3f} "
            f"{ep_conv:>8d} "
            f"{deg_rel:>8.3f} "
            f"{r['avg_exec_time']:>6.3f} "
            f"{100.0 * r['avg_lolp']:>7.3f}"
        )

    # ── paper Table 4 targets for direct comparison ────────────────────────
    print("=" * 118)
    print("\n  Paper Table 4 targets:")
    print(
        f"  {'SA-DRL':<30} {'10.0':>8} {'0.680':>6} {'0.700':>6} "
        f"{'4800':>8} {'1.000':>8} {'0.450':>6} {'4.500':>7}"
    )
    print(
        f"  {'Flat MA-DRL':<30} {'18.0':>8} {'0.770':>6} {'0.790':>6} "
        f"{'3400':>8} {'0.880':>8} {'0.320':>6} {'2.800':>7}"
    )
    print(
        f"  {'HMA-DRL (paper)':<30} {'27.0':>8} {'0.870':>6} {'0.900':>6} "
        f"{'2200':>8} {'0.750':>8} {'0.280':>6} {'1.500':>7}"
    )
    print("=" * 118)


def _convergence_episode(rewards: list, pct: float = 0.95) -> int:
    if not rewards:
        return 0
    final     = np.mean(rewards[-20:]) if len(rewards) >= 20 else (rewards[-1] if rewards else 0)
    threshold = pct * final
    if threshold <= 0:
        return len(rewards)
    for i, r in enumerate(rewards):
        if r >= threshold:
            return i + 1
    return len(rewards)


#________________________________________________

def print_novelty_table(results_fixed: dict, results_arw: dict, 
                        results_scenario: dict):
    """
    Prints the ablation table showing contribution of each novelty.
    Call after running all three HMA variants.
    """
    print("\n" + "=" * 90)
    print("NOVELTY CONTRIBUTION TABLE")
    print(f"{'Variant':<35} {'CostRed%':>8} {'RUR':>6} {'RCIR':>6} "
          f"{'Deg.Idx':>8} {'LOLP%':>7}")
    print("=" * 90)

    baseline_cost = results_fixed["avg_cost_last50"]  # fixed weights as baseline

    for label, r in [
        ("HMA + Fixed Weights (base)",    results_fixed),
        ("HMA + ARW (novelty 1)",         results_arw),
        ("HMA + Scenario-ARW (novelty 2)", results_scenario),
    ]:
        cost_red = max(0.0, 100.0 * (1.0 - r["avg_cost_last50"] / baseline_cost))
        sa_deg   = results_fixed["avg_deg_last50"]
        deg_rel  = r["avg_deg_last50"] / sa_deg if sa_deg > 0 else 1.0
        print(
            f"{label:<35} "
            f"{cost_red:>8.1f} "
            f"{r['avg_rur_last50']:>6.3f} "
            f"{r['rcir']:>6.3f} "
            f"{deg_rel:>8.3f} "
            f"{100.0 * r['avg_lolp']:>7.3f}"
        )
    print("=" * 90)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",    default="all", choices=["all","hma","flat","sa","hma_fixed"])
    parser.add_argument("--episodes",  type=int, default=10000)  
    parser.add_argument("--batch",     type=int, default=256)
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--scenario",  default="normal",
                        choices=["normal","crit_load","pv_outage","dynamic_price","high_res"])
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    methods = ["sa", "flat", "hma"] if args.method == "all" else [args.method]
    results = []

    for m in methods:
        print(f"\n{'='*60}")
        print(f"  Training  {m.upper()}  ({args.episodes} episodes, scenario={args.scenario})")
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
        rule_baseline = compute_rule_based_baseline(n_episodes=50)
        print_comparison_table(results, rule_based_baseline=rule_baseline)
        fixed_r    = next((r for r in results if r["method"] == "hma_fixed"), None)
        arw_r      = next((r for r in results if r["method"] == "hma"), None)
        if fixed_r and arw_r:
            print_novelty_table(fixed_r, arw_r, arw_r)

    out_dir = Path("/home/gkianfar/scratch/Amin/MSH/output/plots")
    plot_all_results(results, out_dir)
    print("\nDone.")
