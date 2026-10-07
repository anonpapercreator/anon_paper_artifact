"""Addendum 2 diagnostic: log every write in one 12-hour replication (E, Poisson).
usage: python3 diag_write_log.py A P REP OUT.npy"""
import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main import load_config, run_replication
import components.filesystem as fsmod
LOG = []
cls = [c for c in vars(fsmod).values() if isinstance(c, type) and hasattr(c, "handle_write") and hasattr(c, "_delete_after")][0]
orig = cls.handle_write
def hw(self, f):
    t0 = self._env.now
    yield from orig(self, f)
    LOG.append((f.created_at, t0, self._env.now, f.size_bytes))
cls.handle_write = hw
a, p, rep = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
cfg = load_config(str(ROOT / "config" / "default_config.yaml"))
cfg["simulation"].update(sim_duration_s=12*3600.0, warmup_s=3600.0, seed=42, n_replications=1)
wl = cfg["workload"]; wl.update(mode="synthetic", read_fraction=0.0, diurnal_enabled=False, iat_distribution="exponential")
wl["lifetime"] = dict(enabled=True, s_inf=0.5, kind="exp", mean_s=1800.0)
cfg["policy_engine"].update(candidate_age_threshold_s=a, cycle_period_s=p)
cfg["adaptive_policy"]["enabled"] = False
cfg["tape_drives"]["stream_zone_writes"] = True
run_replication(cfg, replication_id=rep)
L = np.array(LOG)
np.save(sys.argv[4], L)
c, t0, t1, s = L.T
print("n", len(L), "mean size MB", s.mean()/1e6, "mean iat", np.diff(c).mean(), "mean write time", (t1-c).mean(), "max", (t1-c).max())
for h in range(12):
    m = (c >= h*3600) & (c < (h+1)*3600)
    print(h, m.sum(), round(s[m].sum()/3600/1e6, 1), "MB/s", "wait", round((t1-c)[m].mean(), 3))
