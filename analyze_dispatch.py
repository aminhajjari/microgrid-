"""
analyze_dispatch.py  — run on Narval to produce real EV/BESS power & energy
profiles from a trained checkpoint, plus a dispatch sanity report.

    python analyze_dispatch.py --method hma \
        --ckpt /home/gkianfar/scratch/Amin/MSH/output/checkpoints/hma_weights.pt \
        --out  /home/gkianfar/scratch/Amin/MSH/output/plots

Notes
-----
* The current MicrogridEnv is a SINGLE-NODE power-balance model. There is no
  network, no buses and no voltage state, so a voltage-vs-time plot cannot be
  produced from it (see the write-up). This script therefore covers (1) power
  and (2) energy only, and prints a diagnostic report instead of voltage.
* EV stored energy is NOT tracked by the env (soc_ev is re-drawn at random each
  step and never integrated from p_ev), so EV energy is reported as "not
  modelled" rather than plotted from a phantom state.
"""
import argparse, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from microgrid_env import MicrogridEnv
from hma_drl import HMADRLFramework, HMADRLFixedWeights
try:
    from hma_drl import FlatMADRL, SingleAgentDRL
except Exception:
    FlatMADRL = SingleAgentDRL = None

CAP_BESS = 100.0  # kWh implied by SOC update dSOC = p_bess / (P_BESS_MAX*2)

def build(method, device="cpu"):
    if method in ("hma", "hma_fixed"):
        return (HMADRLFixedWeights if method == "hma_fixed" else HMADRLFramework)(device=device)
    raise SystemExit(f"add your own builder for method={method}")

def load(controller, ckpt):
    w = torch.load(ckpt, map_location="cpu", weights_only=False)
    for name, agent in controller.agents.items():
        agent.actor.load_state_dict(w[f"{name}_actor"])
        agent.critic.load_state_dict(w[f"{name}_critic"])
    controller.supervisor.actor.load_state_dict(w["supervisor_actor"])
    controller.supervisor.critic.load_state_dict(w["supervisor_critic"])
    if "arw" in w:
        controller.arw.load_state_dict(w["arw"])
    print(f"loaded {ckpt}")

def rollout(controller, seed=0):
    env = MicrogridEnv(episode_seed=seed, domain_randomize=False)
    obs, _ = env.reset(); local = np.zeros(4); rows = []; done = False
    while not done:
        controller.arw.is_warming_up = False  # force the trained ARW on for eval
        rw = controller.get_reward_weights(obs)
        action, _ = controller.select_actions(obs, local, explore=False)
        obs, r, done, _, info = env.step(action, reward_weights=rw)
        local = controller.compute_local_rewards(info, rw)
        rows.append((info["p_bess"], info["p_ev"], info["soc_bess"],
                     info["p_grid"], info["p_pv"], info["p_load"], info["load_loss"]))
    a = np.array(rows)
    return dict(p_bess=a[:,0], p_ev=a[:,1], soc=a[:,2], p_grid=a[:,3],
                pv=a[:,4], load=a[:,5], ll=a[:,6])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="hma")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ctrl = build(args.method); load(ctrl, args.ckpt)
    D = rollout(ctrl, args.seed); t = np.arange(len(D["soc"]))

    # ---- Plot 1: power ----
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axhline(0, color="0.6", lw=.8)
    ax.plot(t, -D["p_bess"], "#E4572E", lw=2, marker="o", ms=3, label="BESS power")
    ax.plot(t, -D["p_ev"],  "#2E86AB", lw=2, marker="s", ms=3, label="EV power")
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Power (kW)  [+ charge / − discharge]")
    ax.set_title(f"EV & BESS Power Profiles — {args.method}", fontweight="bold")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(out / f"dispatch_power_{args.method}.png", dpi=130)

    # ---- Plot 2: energy ----
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(t, D["soc"] * CAP_BESS, "#E4572E", lw=2.2, marker="o", ms=3, label="BESS stored energy")
    ax.axhline(0.1 * CAP_BESS, ls="--", color="0.5", lw=1, label="SOC limits")
    ax.axhline(0.9 * CAP_BESS, ls="--", color="0.5", lw=1)
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Stored energy (kWh)")
    ax.set_title(f"BESS Stored-Energy Profile — {args.method}", fontweight="bold")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(out / f"dispatch_energy_{args.method}.png", dpi=130)

    # ---- dispatch report ----
    frac_floor = float((D["soc"] <= 0.1001).mean())
    frac_max   = float((np.abs(D["p_bess"]) >= 49.9).mean())
    true_unmet = np.maximum(0.0, D["p_grid"] - 50.0)        # real loss-of-load
    print("\n================ DISPATCH REPORT ================")
    print(f"SOC range ............. {D['soc'].min():.3f} .. {D['soc'].max():.3f}")
    print(f"SOC swing (Δ) ......... {D['soc'].max()-D['soc'].min():.3f}   (≈0 => battery idle/abused)")
    print(f"steps at SOC floor .... {frac_floor*100:.0f}%   (high => parked at empty)")
    print(f"steps at |P_bess|=max . {frac_max*100:.0f}%   (high => bang-bang, not modulating)")
    print(f"net grid energy ....... {D['p_grid'].sum():.1f} kWh   (<<0 => dumping/export)")
    print(f"TRUE unmet load (>50) . {true_unmet.sum():.1f} kWh over the day")
    print(f"env 'load_loss' (logged as LOLP, fires on OVER-export) = {D['ll'].sum():.1f}")
    print("Voltage: NOT MODELLED — single-node power-balance env, no bus/voltage state.")
    print("=================================================\n")

if __name__ == "__main__":
    main()
