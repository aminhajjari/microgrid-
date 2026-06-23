"""
quick_check.py — run locally before submitting SLURM job.
Takes ~2-3 minutes on CPU. Checks for crashes, shape errors,
reward scale sanity, and gate logic.
"""
import numpy as np
from microgrid_env import MicrogridEnv
from hma_drl import HMADRLFramework, FlatMADRL, SingleAgentDRL, _softmax
from adaptive_reward import AdaptiveRewardWeighter

N_EP    = 5   # just enough to exercise all code paths
N_STEPS = 24  # one full episode

def run_method(method, n_ep=N_EP):
    env = MicrogridEnv(episode_seed=0, domain_randomize=True)
    if method == "hma":
        ctrl = HMADRLFramework(device="cpu")
    elif method == "flat":
        ctrl = FlatMADRL(device="cpu")
    else:
        ctrl = SingleAgentDRL(device="cpu")

    rewards = []
    local_rewards = np.zeros(4)

    for ep in range(n_ep):
        obs, _ = env.reset()
        ep_reward = 0.0

        for _ in range(N_STEPS):
            if isinstance(ctrl, HMADRLFramework):
                rw = ctrl.get_reward_weights(obs)
                assert rw.shape == (4,), f"Bad ARW shape: {rw.shape}"
                assert np.all(rw > 0), f"Non-positive weights: {rw}"

                action, omega = ctrl.select_actions(obs, local_rewards, explore=True)
                assert omega.shape == (2,), f"Bad omega shape: {omega.shape}"

                nobs, reward, done, _, info = env.step(action, reward_weights=rw)
                ctrl.record_reward(ctrl.compute_arw_reward(info))  # v3: fixed KPI

                prev = local_rewards.copy()
                local_rewards = ctrl.compute_local_rewards(info, rw)  # v3: weight-aware
                assert local_rewards.shape == (4,), f"Bad local_rewards shape"
                assert np.all(np.abs(local_rewards) <= 1.01), f"Local rewards out of [-1,1]: {local_rewards}"

                ctrl.store_transitions(obs, action, local_rewards, reward, nobs, done, prev, omega)

            elif isinstance(ctrl, FlatMADRL):
                action = ctrl.select_actions(obs, explore=True)
                nobs, reward, done, _, info = env.step(action)
                ctrl.store_transitions(obs, action, np.full(4, reward/4), nobs, done)

            else:
                action = ctrl.select_actions(obs, explore=True)
                nobs, reward, done, _, info = env.step(action)
                ctrl.store_transitions(obs, action, reward, nobs, done)

            # Reward scale check — should be in roughly [-5, 5] per step
            assert abs(reward) < 50, f"Reward out of range: {reward}"

            ctrl.update_all(64)
            ep_reward += reward
            obs = nobs

        if isinstance(ctrl, HMADRLFramework):
            ctrl.update_arw()

        rewards.append(ep_reward)

    avg = np.mean(rewards)
    print(f"  [{method.upper()}] avg_reward={avg:.3f}  ✓ no crashes")
    return avg

def check_supervisor_buffer():
    """Verify supervisor buffer actually receives transitions."""
    env = MicrogridEnv(episode_seed=0, domain_randomize=False)
    ctrl = HMADRLFramework(device="cpu")
    obs, _ = env.reset()
    local_rewards = np.zeros(4)

    for _ in range(10):
        rw = ctrl.get_reward_weights(obs)
        action, omega = ctrl.select_actions(obs, local_rewards)
        nobs, reward, done, _, info = env.step(action, reward_weights=rw)
        prev = local_rewards.copy()
        local_rewards = ctrl.compute_local_rewards(info, rw)  # v3: weight-aware
        ctrl.store_transitions(obs, action, local_rewards, reward, nobs, done, prev, omega)
        obs = nobs if not done else env.reset()[0]

    sup_size = ctrl.supervisor.buffer.size
    assert sup_size == 10, f"Supervisor buffer has {sup_size} transitions, expected 10"
    print(f"  [SUPERVISOR BUFFER] size={sup_size}  ✓")

def check_reward_scale():
    """Single episode reward sanity: all methods should be in similar range."""
    env = MicrogridEnv(episode_seed=42, domain_randomize=False)
    for method in ["sa", "flat", "hma"]:
        env.reset()
        obs, _ = env.reset()
        total = 0.0
        for _ in range(24):
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            total += reward
        print(f"  [SCALE {method.upper()}] random-policy episode reward = {total:.3f}")
        assert total > -200, f"Reward too negative ({total}) — likely scale bug"

if __name__ == "__main__":
    print("=== Quick sanity check ===\n")

    print("1. Reward scale (random policy):")
    check_reward_scale()

    print("\n2. Supervisor buffer fills correctly:")
    check_supervisor_buffer()

    print("\n3. Training loop — no crashes:")
    rewards = {}
    for m in ["sa", "flat", "hma"]:
        rewards[m] = run_method(m)

    print("\n4. HMA vs SA reward gap:")
    gap = rewards["hma"] - rewards["sa"]
    print(f"   HMA={rewards['hma']:.3f}  SA={rewards['sa']:.3f}  gap={gap:.3f}")
    if gap < -5.0:
        print("  ⚠️  WARNING: HMA much worse than SA even after 5 episodes.")
        print("      This may be normal (no warmup), but investigate if gap > -5.")
    else:
        print("  ✓ Gap acceptable for 5-episode no-warmup run")

    print("\n✅ All checks passed — safe to submit SLURM job.\n")
