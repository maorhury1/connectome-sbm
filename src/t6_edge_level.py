"""
T6 -- Piazza et al.'s analysis, run at the EDGE level.

Their paper fits candidate distributions to node quantities (strength S, degree k, length L,
density rho) and ranks them by KS. Our work operates one level down, on the individual edge
weight w_ij (the synapse count of a single connection), which their analysis never touches.
This runs their exact procedure on w_ij.

Two of their tests are reproduced:

  1. KS ranking     -- MLE fit of each candidate family, ranked by KS statistic.
  2. Rescaling collapse -- their Fig. 2b/e test. Rescale each subpopulation by
                       w_bar = exp((log w - mu)/sigma). If one universal shape underlies all of
                       them, the rescaled distributions collapse onto a single curve. Collapse
                       is quantified by the pairwise two-sample KS between rescaled groups
                       (small = collapses).

Reported for the whole connectome and split by the regimes of T5 (MAP columnar / OPTIC-rest /
NON-OPTIC), all at matched n.

Run from src/:  python t6_edge_level.py
"""
import itertools
import numpy as np
import pandas as pd
from scipy import stats
import config
import data

SCRATCH = str(config.SCRATCH_DIR)
N_DRAW = 15
N_MATCH = 40_000
RNG = np.random.default_rng(0)

FAMILIES = {
    "lognormal":   lambda x: stats.lognorm(*stats.lognorm.fit(x, floc=0)),
    "gamma":       lambda x: stats.gamma(*stats.gamma.fit(x, floc=0)),
    "weibull":     lambda x: stats.weibull_min(*stats.weibull_min.fit(x, floc=0)),
    "exponential": lambda x: stats.expon(*stats.expon.fit(x, floc=0)),
    "powerlaw":    lambda x: stats.pareto(*stats.pareto.fit(x, floc=0)),
    "geometric":   lambda x: stats.geom(min(max(1.0 / max(x.mean(), 1.0), 1e-9), 1 - 1e-9)),
    "poisson":     lambda x: stats.poisson(max(x.mean(), 1e-9)),
}


def ks_all(x):
    out = {}
    for name, fit in FAMILIES.items():
        try:
            out[name] = stats.kstest(x, fit(x).cdf).statistic
        except Exception:
            out[name] = np.nan
    return out


def main():
    lab = data.load_labels()
    # CRITICAL: connections_princeton has ONE ROW PER (pre, post, NEUROPIL). A row is a fragment
    # of a connection, not a connection. Summing over neuropils is what gives the actual edge
    # weight -- and once summed, the minimum is 5, because the codex table only reports pairs
    # reaching 5 synapses in total. Treating rows as edges invents ~1.6M sub-5 "edges" that do
    # not exist and corrupts every fit below.
    raw = pd.read_parquet(f"{SCRATCH}/conn.parquet")
    c = (raw.groupby(["pre_root_id", "post_root_id"], as_index=False).syn_count.sum())
    print(f"rows in table: {len(raw):,}  ->  real edges after summing neuropils: {len(c):,}")
    assert c.syn_count.min() >= 5, f"expected w>=5 after summing, got {c.syn_count.min()}"
    hexids = set(pd.read_csv(f"{SCRATCH}/column_assignment.csv.gz").root_id.tolist())
    optic = set(lab.index[lab.super_class == "optic"].tolist())

    def regime(ids):
        return np.where(pd.Series(ids).isin(hexids), "MAP (columnar)",
                 np.where(pd.Series(ids).isin(optic), "OPTIC-rest", "NON-OPTIC"))

    # an edge is assigned to a regime when BOTH endpoints are in it (unambiguous edges only)
    rp, rq = regime(c.pre_root_id.values), regime(c.post_root_id.values)
    c = c.assign(regime=np.where(rp == rq, rp, "mixed"))
    w_all = c.syn_count.values.astype(float)
    print(f"edges: {len(c):,}   w range [{w_all.min():.0f}, {w_all.max():.0f}]   median {np.median(w_all):.0f}")
    print(c.regime.value_counts().to_string(), "\n")

    # ---- 1. KS ranking, all edges and per regime, at matched n ----
    rows = []
    groups = [("ALL EDGES", w_all)] + [(r, g.syn_count.values.astype(float))
                                       for r, g in c.groupby("regime") if r != "mixed"]
    for name, v in groups:
        for _ in range(N_DRAW):
            x = RNG.choice(v, min(N_MATCH, len(v)), replace=False)
            rows.append(dict(group=name, n=len(x), **ks_all(x)))
    r = pd.DataFrame(rows)
    r.to_csv(f"{SCRATCH}/t6_edge_ks.csv", index=False)
    fam = list(FAMILIES)
    t = r.groupby("group")[fam].mean()
    t["WINNER"] = t.idxmin(axis=1)
    t["lognormal_rank"] = t[fam].rank(axis=1).lognormal.astype(int)
    print("=" * 104)
    print("1. KS on EDGE WEIGHTS w_ij (their procedure, one level down). Lower = better.")
    print("=" * 104)
    print(t.round(4).to_string())

    # ---- 2. their rescaling-collapse test, across cell-type pairs ----
    print("\n" + "=" * 104)
    print("2. Rescaling collapse (their Fig. 2b/e): does one universal shape underlie all groups?")
    print("   two-sample KS between RESCALED groups; small = collapses onto a common curve")
    print("=" * 104)
    pt = lab.primary_type
    c2 = c.assign(t_pre=pt.reindex(c.pre_root_id).values)
    big = c2.groupby("t_pre").size().sort_values(ascending=False).head(12).index
    for scale in ["log (their rescaling)", "raw"]:
        resc = {}
        for tname in big:
            v = c2.loc[c2.t_pre == tname, "syn_count"].values.astype(float)
            y = np.log(v) if scale.startswith("log") else v
            resc[tname] = (y - y.mean()) / y.std()
        ks = [stats.ks_2samp(resc[a], resc[b]).statistic for a, b in itertools.combinations(big, 2)]
        print(f"  {scale:24s} mean pairwise KS = {np.mean(ks):.4f}   median = {np.median(ks):.4f}"
              f"   max = {np.max(ks):.4f}   ({len(big)} largest cell types)")


if __name__ == "__main__":
    main()
