"""
Gated evaluation of the overnight fits (CP-3 comparison table).

For every saved fit under WORK_DIR/results it computes, against the withheld label
hierarchy, homogeneity / completeness / V / AMI / ARI (per level), plus block count,
MDL, and cross-seed partition stability (pairwise AMI between seeds). Results are
aggregated per (model, direction, degree-correction) and appended to RESULTS.md.

IMPORTANT SCOPE (recorded, not hidden): these fits are the >=5-synapse, FLAT regime,
which the plan calls the *sensitivity* condition. The *canonical* setup (>=1 synapse,
nested) has not been run, and MDL is only comparable *within* a weight family (Sec 3.5).
So this table selects the top candidates to carry into E2b; it is not the final word.

This is a GATED script: it evaluates, writes the table, and STOPS for sign-off.
Run from src/:  python eval_sweep.py
"""
import re
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import (homogeneity_completeness_v_measure,
                             adjusted_mutual_info_score, adjusted_rand_score)
import config

RESULTS = config.WORK_DIR / "results"
RUN_RE = re.compile(r"(?P<model>[a-z]+)_t(?P<thr>\d+)_(?P<dir>dir|und)_(?P<dc>dc|ndc)_s(?P<seed>\d+)$")


def load_labels():
    """neuron -> label hierarchy, evaluation-only (never seen by the fits)."""
    cls = pd.read_csv(config.CLASSIFICATION,
                      usecols=["root_id", "super_class", "class", "sub_class", "side"])
    ct = pd.read_csv(config.CELL_TYPES, usecols=["root_id", "primary_type"])
    lab = cls.merge(ct, on="root_id", how="left").set_index("root_id")
    return lab


def score_fit(run_dir, lab, levels):
    d = np.load(run_dir / "partition.npz")
    s = pd.Series(d["blocks"], index=d["node_ids"])
    out = {"n_blocks": int(s.nunique())}
    for lvl in levels:
        y = lab[lvl].reindex(s.index)
        keep = y.notna().to_numpy() & (y.to_numpy().astype(str) != "")
        if keep.sum() < 10:
            continue
        yt = y.to_numpy()[keep].astype(str)
        b = s.to_numpy()[keep]
        h, c, v = homogeneity_completeness_v_measure(yt, b)
        out[f"h_{lvl}"], out[f"c_{lvl}"], out[f"v_{lvl}"] = h, c, v
        if lvl == "primary_type":                       # AMI/ARI only where reported (speed)
            out[f"ami_{lvl}"] = adjusted_mutual_info_score(yt, b)
            out[f"ari_{lvl}"] = adjusted_rand_score(yt, b)
    import json
    out.update(json.loads((run_dir / "summary.json").read_text())["config"])
    out["mdl"] = json.loads((run_dir / "summary.json").read_text())["mdl_entropy"]
    return out


def stability(partitions, max_nodes=30000, seed=0):
    """mean pairwise AMI between seed partitions (estimated on a node subsample)."""
    if len(partitions) < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    amis = []
    for a, b in itertools.combinations(partitions, 2):
        common = a.index.intersection(b.index)
        if len(common) > max_nodes:
            common = common[rng.choice(len(common), max_nodes, replace=False)]
        amis.append(adjusted_mutual_info_score(a.loc[common].to_numpy(),
                                               b.loc[common].to_numpy()))
    return float(np.mean(amis))


def main():
    import time
    lab = load_labels()
    levels = config.LABEL_LEVELS
    rows, parts_by_cfg = [], defaultdict(list)
    # use the >=5 fits (the >=1 fits are byte-identical: source pre-floored at 5)
    run_dirs = [d for d in sorted(RESULTS.glob("*_t5_*"))
                if RUN_RE.match(d.name) and (d / "partition.npz").exists()]
    n = len(run_dirs)
    print(f"[eval] scoring {n} fits x {len(levels)} label levels ...", flush=True)
    t0 = time.time()
    for i, run_dir in enumerate(run_dirs, 1):
        m = RUN_RE.match(run_dir.name)
        r = score_fit(run_dir, lab, levels)
        r.update(model=m["model"], dir=m["dir"], dc=m["dc"], seed=int(m["seed"]))
        rows.append(r)
        d = np.load(run_dir / "partition.npz")
        parts_by_cfg[(m["model"], m["dir"], m["dc"])].append(
            pd.Series(d["blocks"], index=d["node_ids"]))
        print(f"  [{i}/{n}] {run_dir.name}  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    print(f"[eval] scoring done in {time.time()-t0:.0f}s; computing stability ...", flush=True)

    metric_cols = ["n_blocks", "mdl"] + [f"v_{l}" for l in levels] \
        + ["h_primary_type", "c_primary_type", "ami_primary_type", "ari_primary_type"]
    agg = df.groupby(["model", "dir", "dc"])[metric_cols].mean().reset_index()
    print(f"[eval] computing seed-stability for {len(agg)} configs (30k-node subsample) ...", flush=True)
    t1, stab = time.time(), []
    for j, r in enumerate(agg.itertuples(), 1):
        stab.append(stability(parts_by_cfg[(r.model, r.dir, r.dc)]))
        print(f"  stability [{j}/{len(agg)}] {r.model}-{r.dir}-{r.dc}  ({time.time()-t1:.0f}s)", flush=True)
    agg["stability_ami"] = stab
    agg = agg.sort_values("v_primary_type", ascending=False).round(3)

    # ---- console ----
    pd.set_option("display.width", 220, "display.max_columns", 40)
    show = ["model", "dir", "dc", "n_blocks", "v_super_class", "v_class",
            "v_sub_class", "v_primary_type", "h_primary_type", "c_primary_type",
            "ami_primary_type", "stability_ami", "mdl"]
    print("CP-3 sweep evaluation (>=5, flat = SENSITIVITY regime; mean over 5 seeds):\n")
    print(agg[show].to_string(index=False))

    # ---- append aggregated form to RESULTS.md ----
    md = ["\n\n## CP-3 — Overnight sweep evaluation (aggregated)\n",
          "*Scope: >=5-synapse, FLAT fits (the plan's **sensitivity** regime, not canonical "
          ">=1/nested). MDL comparable only within a weight family. Mean over 5 seeds; "
          "stability = mean pairwise AMI between seeds (30k-node subsample). "
          "Ranked by V(primary_type).*\n",
          "| model | dir | dc | blocks | V super | V class | V subcl | V type | homog(type) | compl(type) | AMI(type) | stability | MDL |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in agg[show].itertuples(index=False):
        md.append("| " + " | ".join(
            [str(r.model), r.dir, r.dc, f"{r.n_blocks:.0f}",
             f"{r.v_super_class:.3f}", f"{r.v_class:.3f}", f"{r.v_sub_class:.3f}",
             f"{r.v_primary_type:.3f}", f"{r.h_primary_type:.3f}", f"{r.c_primary_type:.3f}",
             f"{r.ami_primary_type:.3f}", f"{r.stability_ami:.3f}", f"{r.mdl:.3e}"]) + " |")
    (config.REPO_DIR / "RESULTS.md").open("a").write("\n".join(md) + "\n")
    print("\n[gated] wrote aggregated table to RESULTS.md. STOP — awaiting sign-off before E2b.")


if __name__ == "__main__":
    main()
