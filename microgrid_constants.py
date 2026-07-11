DT = 1.0
GAMMA = 0.4   # FIX-17: was 0.05 — 10-25x smaller than r_cost's typical magnitude,
              # so wear cost never outweighed grid-cost savings and the agent
              # learned to dump the battery immediately regardless of tariff.
              # 0.4 puts max-power wear cost (~0.4) on par with a typical
              # discharge's cost saving (~0.4-1.3), creating a real trade-off.
P_BESS_MAX = 50.0
KAPPA = 1.4
ZETA = 0.1
