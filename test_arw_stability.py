import numpy as np
from adaptive_reward import AdaptiveRewardWeighter, BASE_WEIGHTS

OBS = np.zeros(7, dtype=np.float32)
rng = np.random.default_rng(0)

def mean_weights(arw, n=300):
    return np.mean([arw.get_weights(OBS) for _ in range(n)], axis=0)

# Test A: pure-noise reward -> weights must NOT drift from init
arw = AdaptiveRewardWeighter(warmup_episodes=0)
for ep in range(80):
    for t in range(24):
        arw.get_weights(OBS)
        arw.record(-3.0 + rng.standard_normal())
    info = arw.update()
w = mean_weights(arw)
drift = np.abs(w - BASE_WEIGHTS).max()
print(f"[A] baseline={info['arw_baseline']:.2f}  loss={info['arw_loss']:.3f}")
print(f"[A] mean weights: {np.round(w,3)}  (init {BASE_WEIGHTS})  max drift={drift:.3f}")
assert -6.0 < info["arw_baseline"] < -1.0, "FAIL: baseline not in raw per-step reward units - FIX-8/10 missing?"
assert drift < 0.15, f"FAIL: drifted {drift:.3f} under pure noise"
print("[A] PASS - no drift under noise\n")

# Test B: reward favoring w_s -> mean w_s must rise
arw = AdaptiveRewardWeighter(warmup_episodes=0)
ws0 = mean_weights(arw)[3]
for ep in range(150):
    for t in range(24):
        w_t = arw.get_weights(OBS)
        arw.record(-5.0 + 2.0 * w_t[3] + 0.3 * rng.standard_normal())
    arw.update()
ws1 = mean_weights(arw)[3]
print(f"[B] mean w_s: {ws0:.3f} -> {ws1:.3f}")
assert ws1 > ws0 + 0.05, "FAIL: ARW did not follow a clear reward signal"
print("[B] PASS - ARW learns from a signed advantage")
print("\nAll ARW sanity checks passed")
