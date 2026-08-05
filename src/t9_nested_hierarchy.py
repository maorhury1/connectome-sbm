"""
t9: Does the SBM hierarchy level correspond to the biological label level?

Reviewer objection: "your ~190-block solution isn't cell types, it failed."
Counter-hypothesis: the nested SBM has SEVERAL levels, and different levels should line
up with different biological levels (super_class -> class -> sub_class -> primary_type).
If so, a resolution mismatch at one level is a *result*, not a failure.

Method
------
For every nested run (model x seed, degree-corrected + directed, threshold>=5) we take the
per-neuron partition at every level of the fitted hierarchy and score it against each of
the four biological levels with ADJUSTED mutual information (chance-corrected: mandatory
here, because block counts run 2 -> 5000 and label counts 10 -> 8600, and plain NMI/MI
rewards granularity outright).

Scoring is done per hemisphere (side in {left,right}) separately -- the two hemispheres are
near-duplicates, so pooling them would inflate every score identically -- and averaged over
seeds.

Data caveat handled here
------------------------
The `level_*` arrays saved by nested_sweep.py (from `state.project_level(l)`) are CORRUPT:
their block counts do not match the `levels` field of the run's JSON, they are not nested,
and `level_0` has AMI ~0 against everything. The raw `bs_*` tree in the same file IS valid.
We therefore rebuild each level by composing the bs maps:
    level_0 = bs_0[node];  level_{l+1} = bs_{l+1}[level_l]
and verify at runtime that the resulting chain is strictly nested and its block counts match
the JSON `levels` field. INTEGRITY section of the output reports this.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, homogeneity_completeness_v_measure

import config
import data

NESTED_DIR = "/var/tmp/csbm_work/nested_results"
PATTERN = "*_dc_dir_s*_partition.npz"
BIO_LEVELS = config.LABEL_LEVELS            # super_class, class, sub_class, primary_type
SIDES = ["left", "right"]
OUT_CSV = os.path.join(str(config.SCRATCH_DIR), "t9_nested_hierarchy_rows.csv")


# --------------------------------------------------------------------------- loading
def build_chain(npz):
    """Per-neuron partition at every level, rebuilt from the raw bs_* tree.

    Returns list of arrays (length N), index 0 = finest. Trivial 1-block levels dropped.
    """
    cur = np.asarray(npz["bs_0"], dtype=np.int64)
    chain = [cur.copy()]
    l = 1
    while f"bs_{l}" in npz:
        m = np.asarray(npz[f"bs_{l}"], dtype=np.int64)
        if cur.max() >= len(m):                       # tree ended / padding mismatch
            break
        cur = m[cur]
        chain.append(cur.copy())
        l += 1
    # drop trailing trivial levels (everything in one block: AMI undefined/0 by construction)
    out = [c for c in chain if len(np.unique(c)) > 1]
    return out


def is_nested(chain):
    """True if each level is a strict coarsening of the level below it."""
    for a, b in zip(chain[:-1], chain[1:]):
        order = np.argsort(a, kind="stable")
        a_s, b_s = a[order], b[order]
        bounds = np.flatnonzero(np.diff(a_s)) + 1
        for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(a_s)]):
            if len(np.unique(b_s[lo:hi])) != 1:
                return False
    return True


def discover_runs():
    """(model, seed_tag, partition_path, json_levels) for every SUCCESSFUL dc_dir nested run."""
    runs = []
    skipped = []
    for p in sorted(glob.glob(os.path.join(NESTED_DIR, PATTERN))):
        base = p[: -len("_partition.npz")]
        jpath = base + ".json"
        if not os.path.exists(jpath):
            skipped.append((os.path.basename(base), "no json"))
            continue
        meta = json.load(open(jpath))
        if "levels" not in meta or meta.get("status") == "FAILED":
            skipped.append((os.path.basename(base), meta.get("reason", "no levels in json")))
            continue
        try:
            npz = np.load(p)
        except Exception as e:
            skipped.append((os.path.basename(base), f"unreadable: {e}"))
            continue
        runs.append(dict(model=meta["model"], seed=meta["seed"],
                         tag=os.path.basename(base), path=p,
                         json_levels=meta["levels"], npz=npz))
    return runs, skipped


# --------------------------------------------------------------------------- scoring
def score_run(run, labels_by_node, integrity):
    chain = build_chain(run["npz"])
    jl = [b for b in run["json_levels"] if b > 1]
    ok_counts = [len(np.unique(c)) for c in chain] == jl
    ok_nested = is_nested(chain)
    # what the (corrupt) level_* arrays would have said, for the record
    corrupt_ami = None
    if "level_0" in run["npz"]:
        corrupt_ami = adjusted_mutual_info_score(run["npz"]["level_0"], chain[0])
    integrity.append(dict(tag=run["tag"], n_levels=len(chain),
                          counts_match_json=ok_counts, strictly_nested=ok_nested,
                          ami_level0_vs_bs0=corrupt_ami))
    if not (ok_counts and ok_nested):
        print(f"[warn] {run['tag']}: integrity check failed "
              f"(counts_match={ok_counts}, nested={ok_nested}) -- SKIPPED", flush=True)
        return []

    rows = []
    for lev, blocks in enumerate(chain):
        for side in SIDES:
            side_m = (labels_by_node["side"].values == side)
            for bio in BIO_LEVELS:
                lab = labels_by_node[bio].values
                m = side_m & pd.notna(lab)
                b_, l_ = blocks[m], lab[m].astype(str)
                ami = adjusted_mutual_info_score(b_, l_)
                h, c, _ = homogeneity_completeness_v_measure(l_, b_)  # true=labels, pred=blocks
                rows.append(dict(model=run["model"], seed=run["seed"], tag=run["tag"],
                                 level=lev, side=side, bio=bio,
                                 n_neurons=int(m.sum()),
                                 n_blocks=int(len(np.unique(b_))),
                                 n_labels=int(len(np.unique(l_))),
                                 ami=ami, homogeneity=h, completeness=c))
    return rows


def null_control(run, labels_by_node, rng):
    """AMI of a size-matched RANDOM partition against primary_type: must be ~0."""
    chain = build_chain(run["npz"])
    side_m = (labels_by_node["side"].values == "left")
    lab = labels_by_node["primary_type"].values
    m = side_m & pd.notna(lab)
    out = []
    for lev in (0, min(2, len(chain) - 1)):
        b = chain[lev][m].copy()
        rng.shuffle(b)
        out.append((lev, int(len(np.unique(b))),
                    adjusted_mutual_info_score(b, lab[m].astype(str))))
    return out


# --------------------------------------------------------------------------- reporting
def main():
    t0 = time.time()
    if "--from-csv" in sys.argv:                       # re-print the report, no rescoring
        report(pd.read_csv(OUT_CSV))
        return
    runs, skipped = discover_runs()
    print(f"[t9] {len(runs)} successful nested dc_dir runs; {len(skipped)} skipped")
    for tag, why in skipped:
        print(f"      skip {tag}: {why}")
    if not runs:
        sys.exit("no runs found")

    node_ids = runs[0]["npz"]["node_ids"]
    for r in runs:
        assert np.array_equal(r["npz"]["node_ids"], node_ids), f"node_ids differ: {r['tag']}"
    labels = data.load_labels().reindex(node_ids)
    print(f"[t9] {len(node_ids):,} neurons; "
          + ", ".join(f"{s}={int((labels['side']==s).sum()):,}" for s in SIDES))

    # label-count sanity table (per hemisphere, restricted to labelled neurons)
    print("\n=== BIOLOGICAL LEVEL SIZES (per hemisphere, labelled neurons only) ===")
    print(f"{'bio level':<14}{'left n':>9}{'left #labels':>14}{'right n':>9}{'right #labels':>15}")
    for bio in BIO_LEVELS:
        cells = []
        for side in SIDES:
            m = (labels["side"].values == side) & pd.notna(labels[bio].values)
            cells += [int(m.sum()), int(len(np.unique(labels[bio].values[m].astype(str))))]
        print(f"{bio:<14}{cells[0]:>9,}{cells[1]:>14,}{cells[2]:>9,}{cells[3]:>15,}")

    integrity, rows = [], []
    for r in runs:
        t = time.time()
        rows += score_run(r, labels, integrity)
        print(f"[t9] scored {r['tag']} ({time.time()-t:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("\n=== INTEGRITY ===")
    print(f"{'run':<34}{'levels':>7}{'counts==json':>14}{'nested':>8}{'AMI(level_0,bs_0)':>19}")
    for i in integrity:
        a = "n/a" if i["ami_level0_vs_bs0"] is None else f"{i['ami_level0_vs_bs0']:.4f}"
        print(f"{i['tag']:<34}{i['n_levels']:>7}{str(i['counts_match_json']):>14}"
              f"{str(i['strictly_nested']):>8}{a:>19}")

    rng = np.random.default_rng(0)
    print("\n=== NULL CONTROL (shuffled blocks vs primary_type, left hemisphere) ===")
    for r in runs[:3]:
        for lev, nb, a in null_control(r, labels, rng):
            print(f"  {r['tag']:<34} level {lev}  {nb:>5} blocks  AMI={a:+.5f}")

    report(df)
    print(f"\n[t9] rows -> {OUT_CSV}")
    print(f"[t9] done in {time.time()-t0:.0f}s")


def report(df):
    # ---- main table: model x level, averaged over seeds and hemispheres
    print("\n=== SBM LEVEL  vs  BIOLOGICAL LEVEL (AMI, mean over seeds and hemispheres) ===")
    hdr = (f"{'model':<12}{'lvl':>4}{'seeds':>6}{'blocks':>9}"
           + "".join(f"{b[:11]:>12}" for b in BIO_LEVELS)
           + f"{'  best':<14}{'AMI':>7}")
    print(hdr)
    print("-" * len(hdr))
    summary = []
    for model, gm in df.groupby("model"):
        for lev, gl in gm.groupby("level"):
            per_bio = {b: gl[gl.bio == b]["ami"].mean() for b in BIO_LEVELS}
            nb = gl[gl.bio == BIO_LEVELS[0]]["n_blocks"].mean()
            nseed = gl["seed"].nunique()
            best = max(per_bio, key=per_bio.get)
            print(f"{model:<12}{lev:>4}{nseed:>6}{nb:>9.0f}"
                  + "".join(f"{per_bio[b]:>12.3f}" for b in BIO_LEVELS)
                  + f"  {best:<12}{per_bio[best]:>7.3f}")
            summary.append(dict(model=model, level=lev, n_seeds=nseed, n_blocks=nb,
                                best=best, best_ami=per_bio[best], **per_bio))
        print()
    sm = pd.DataFrame(summary)
    sm.to_csv(OUT_CSV.replace("_rows.csv", "_summary.csv"), index=False)

    # ---- direction of the mismatch: are blocks pure (homogeneity) or complete?
    print("=== HOMOGENEITY / COMPLETENESS of blocks w.r.t. primary_type "
          "(mean over seeds+hemispheres) ===")
    print(f"{'model':<12}{'lvl':>4}{'blocks':>9}{'homog':>8}{'compl':>8}")
    g = df[df.bio == "primary_type"].groupby(["model", "level"])
    for (model, lev), gg in g:
        print(f"{model:<12}{lev:>4}{gg['n_blocks'].mean():>9.0f}"
              f"{gg['homogeneity'].mean():>8.3f}{gg['completeness'].mean():>8.3f}")

    # ---- the ladder: which bio level wins, as a function of how many blocks the level has
    #      (pooled over every model x seed x level, so it does not depend on level INDEX,
    #       which is not comparable across runs of different depth)
    print("\n=== WHICH BIO LEVEL WINS, BY BLOCK COUNT (every model x seed x level) ===")
    nb = df.groupby(["model", "seed", "level"])["n_blocks"].mean().reset_index()
    per = df.groupby(["model", "seed", "level", "bio"])["ami"].mean().reset_index()
    best = per.loc[per.groupby(["model", "seed", "level"])["ami"].idxmax()].merge(
        nb, on=["model", "seed", "level"])
    best = best[best.n_blocks >= 3]
    best["bin"] = pd.cut(best.n_blocks, [3, 10, 30, 100, 300, 1000, 10000])
    ct = pd.crosstab(best["bin"], best["bio"]).reindex(columns=BIO_LEVELS, fill_value=0)
    print(ct.to_string())

    # ---- per-hemisphere consistency check (left vs right should agree)
    print("\n=== HEMISPHERE CONSISTENCY (|AMI_left - AMI_right|) ===")
    piv = df.pivot_table(index=["model", "seed", "level", "bio"], columns="side",
                         values="ami").reset_index()
    piv["d"] = (piv["left"] - piv["right"]).abs()
    piv = piv.merge(nb, on=["model", "seed", "level"])
    print(f"  all levels:        max {piv['d'].max():.3f}, mean {piv['d'].mean():.3f}")
    big = piv[piv.n_blocks >= 10]
    print(f"  levels >=10 blocks: max {big['d'].max():.3f}, mean {big['d'].mean():.3f}")
    print("  (large left/right gaps are confined to the 2-4 block top of the tree, where the "
          "root split is hemisphere-asymmetric)")


if __name__ == "__main__":
    main()
