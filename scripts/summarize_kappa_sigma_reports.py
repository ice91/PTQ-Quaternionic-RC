#!/usr/bin/env python
import sys, json, glob, os
import pandas as pd

def pick(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

rows = []
paths = sys.argv[1:] if len(sys.argv) > 1 else glob.glob("out/robustness/report_*.json")
for p in sorted(paths):
    with open(p, "r", encoding="utf-8") as f:
        J = json.load(f)
    m = J.get("multivar", {})
    if not m:
        continue
    tag = os.path.splitext(os.path.basename(p))[0].replace("report_", "")
    beta = pick(m, ["kappa_sigma","beta"], [None,None,None])
    se   = pick(m, ["kappa_sigma","se"],   [None,None,None])
    R2   = pick(m, ["kappa_sigma","R2"])
    AICc = pick(m, ["kappa_sigma","AICc"])
    dAIC = pick(m, ["delta_AICc"], {})
    dist = pick(m, ["distance_invariant"], {})
    rows.append(dict(
        tag=tag,
        a=beta[0], a_se=se[0],
        b_kappa=beta[1], b_se=se[1],
        c_sigma=beta[2], c_se=se[2],
        R2=R2, AICc=AICc,
        dAIC_kappa_only=dAIC.get("kappa_only"),
        dAIC_sigma_only=dAIC.get("sigma_only"),
        dAIC_kappa_sigma=dAIC.get("kappa_sigma", 0.0),
        dist_n=dist.get("n"), dist_b_kappa=pick(dist, ["beta"], [None,None,None])[1] if dist else None,
        dist_c_sigma=pick(dist, ["beta"], [None,None,None])[2] if dist else None,
        dist_R2=dist.get("R2"),
    ))

df = pd.DataFrame(rows).sort_values("tag")
df.to_csv(sys.stdout, index=False)
