"""
T1 -- is the lognormal connectome a mixture artefact?

Piazza et al. (bioRxiv 2025.02.27.640551) fit lognormals to the MARGINAL distributions of node
strength S and degree k pooled over every neuron in a connectome, and select the lognormal by
its low KS statistic. A lognormal marginal can, however, be produced by MIXING many
non-lognormal groups with different means. Their Galton-Watson derivation predicts lognormality
per neuron, so it must survive conditioning on a homogeneous group -- if it only appears after
pooling, the law is an aggregation effect rather than a property of neurons.

So we fit the lognormal three ways and compare:
    ALL      -- every neuron pooled (their setting)
    by TYPE  -- within each annotated cell type
    by BLOCK -- within each SBM block (lognormal, directed, degree-corrected)

The raw KS statistic CANNOT be compared across these, because KS shrinks with sample size
(critical value ~ 1.36/sqrt(n)) and the groups are far smaller than the pool. Each KS is
therefore divided by its own parametric-bootstrap expectation: resample n points from the
fitted lognormal, refit, recompute KS. That null absorbs the sample-size effect, so

    ks_ratio = KS_observed / E[KS | data really were lognormal]

    ~1  -> indistinguishable from lognormal at that sample size
    >1  -> worse than lognormal

Run from src/:  python t1_mixture_test.py
"""
import glob
import numpy as np
import pandas as pd
from scipy import stats
import config
import data

SCRATCH = str(config.SCRATCH_DIR)
N_BOOT = 60
MIN_GROUP = 50
RNG = np.random.default_rng(0)


def ks_ratio(x, rng):
    """KS of a lognormal MLE fit, divided by its parametric-bootstrap expectation."""
    x = np.asarray(x, float)
    x = x[x > 0]
    if len(x) < MIN_GROUP:
        return None
    lg = np.log(x)
    mu, sd = lg.mean(), lg.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return None
    obs = stats.kstest(lg, "norm", args=(mu, sd)).statistic
    null = []
    for _ in range(N_BOOT):
        y = rng.normal(mu, sd, len(x))
        null.append(stats.kstest(y, "norm", args=(y.mean(), y.std(ddof=0))).statistic)
    null = np.array(null)
    return dict(n=len(x), ks=obs, ks_null=null.mean(),
                ratio=obs / null.mean(), p=float((null >= obs).mean()))


PER_GROUP = []


def summarise(name, quantity, groups, rng):
    """Summary across groups.

    Two different things are reported, because they answer different questions:

      KS   -- the sup-norm DISTANCE between the data and its fitted lognormal. This is a
              measure of shape, and is comparable across sample sizes: it converges to the
              true deviation as n grows. This is the number to compare across levels.
      ratio-- KS divided by its bootstrap null. This measures DETECTABILITY, and for any
              fixed shape deviation d it grows like d*sqrt(n) (KS_obs -> d, KS_null ~ 1/sqrt(n)).
              So it must NOT be compared between a 138k pool and a 150-neuron group; it is
              only meaningful as "is this group distinguishable from lognormal at its own n".
    """
    rows = []
    for g, v in groups:
        r = ks_ratio(v, rng)
        if r:
            r.update(group=str(g), level=name, quantity=quantity)
            rows.append(r)
    if not rows:
        return None
    d = pd.DataFrame(rows)
    PER_GROUP.append(d)
    w = d.n / d.n.sum()
    return dict(level=name, quantity=quantity, n_groups=len(d), n_neurons=int(d.n.sum()),
                median_n=int(d.n.median()),
                KS_wmean=float((d.ks * w).sum()), KS_median=float(d.ks.median()),
                ks_ratio_wmean=float((d.ratio * w).sum()), ks_ratio_median=float(d.ratio.median()),
                pct_groups_consistent=float(100 * (d.p > 0.05).mean()))


def main():
    # Piazza et al. use ALL reported synapses, so read the unthresholded connection table
    # (the project's cached edge file is thresholded at w>=5 and would undercount both S and k).
    c = pd.read_parquet(f"{SCRATCH}/conn.parquet")
    lab = data.load_labels()
    print(f"connections: {len(c):,} (syn_count {c.syn_count.min()}-{c.syn_count.max()})")

    ends = pd.concat([c.rename(columns={"pre_root_id": "id", "post_root_id": "other"}),
                      c.rename(columns={"post_root_id": "id", "pre_root_id": "other"})])
    S = ends.groupby("id").syn_count.sum().astype(float)     # node strength = total synapses
    k = ends.groupby("id").other.nunique().astype(float)     # degree = distinct partners
    print(f"neurons: {len(S):,}   median S={S.median():.0f}   median k={k.median():.0f}")

    z = np.load(sorted(glob.glob(f"{config.WORK_DIR}/results/lognormal_t5_dir_dc_s0/partition.npz"))[0])
    blk = pd.Series(z["blocks"], index=z["node_ids"])

    out = []
    for qname, q in [("strength S", S), ("degree k", k)]:
        df = pd.DataFrame({"v": q})
        df["type"] = lab.primary_type.reindex(df.index)
        df["blk"] = blk.reindex(df.index)
        out.append(summarise("ALL (pooled)", qname, [("all", df.v.values)], RNG))
        out.append(summarise("by TYPE", qname,
                             [(g, d.v.values) for g, d in df.dropna(subset=["type"]).groupby("type")
                              if len(d) >= MIN_GROUP], RNG))
        out.append(summarise("by BLOCK", qname,
                             [(g, d.v.values) for g, d in df.dropna(subset=["blk"]).groupby("blk")
                              if len(d) >= MIN_GROUP], RNG))

    r = pd.DataFrame([o for o in out if o])
    r.to_csv(f"{SCRATCH}/t1_mixture.csv", index=False)
    pg = pd.concat(PER_GROUP)
    pg.to_csv(f"{SCRATCH}/t1_per_group.csv", index=False)

    print("\n" + "=" * 104)
    print("KS       = distance from the fitted lognormal. COMPARABLE across levels. Lower = more lognormal.")
    print("ks_ratio = KS / bootstrap null. Detectability only; grows ~sqrt(n), NOT comparable across levels.")
    print("=" * 104)
    print(r.round(3).to_string(index=False))

    # Does the ratio simply track sample size? (it should, if shape deviation is fixed)
    print("\n--- ratio vs n, within levels (confirms ratio is an n-artefact) ---")
    for (lv, q), g in pg.groupby(["level", "quantity"]):
        if len(g) > 10:
            print(f"  {lv:14s} {q:11s} corr(log n, ratio)={np.corrcoef(np.log(g.n), g.ratio)[0,1]:+.2f}"
                  f"   corr(log n, KS)={np.corrcoef(np.log(g.n), g.ks)[0,1]:+.2f}")

    # THE DECISIVE COMPARISON. Raw KS has a noise floor that grows as n shrinks, so a 150-neuron
    # group cannot be compared to a 138k pool directly. Instead, for every group we draw a random
    # subsample OF THE SAME SIZE from the pooled data. Both then carry the identical noise floor,
    # and the only difference left is whether conditioning on a homogeneous group makes the
    # distribution more or less lognormal.
    print("\n" + "=" * 104)
    print("MATCHED-n: each group vs a random pooled subsample of identical size")
    print("  KS_group < KS_pooled  -> conditioning IMPROVES lognormality (law holds per group)")
    print("  KS_group > KS_pooled  -> conditioning DEGRADES it (pooled lognormal is a mixture effect)")
    print("=" * 104)
    for qname, q in [("strength S", S), ("degree k", k)]:
        pool = np.log(q.values[q.values > 0])
        for lv in ["by TYPE", "by BLOCK"]:
            g = pg[(pg.level == lv) & (pg.quantity == qname)]
            if not len(g):
                continue
            gk, pk = [], []
            for n, ks in zip(g.n.values, g.ks.values):
                y = RNG.choice(pool, int(n), replace=False)
                pk.append(stats.kstest(y, "norm", args=(y.mean(), y.std(ddof=0))).statistic)
                gk.append(ks)
            gk, pk = np.array(gk), np.array(pk)
            print(f"  {qname:11s} {lv:9s}  KS_group={gk.mean():.4f}   KS_pooled_matched={pk.mean():.4f}"
                  f"   group worse in {100*(gk>pk).mean():.0f}% of {len(gk)} groups")


if __name__ == "__main__":
    main()
