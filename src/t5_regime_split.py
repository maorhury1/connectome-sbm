"""
T5 -- does the lognormal connectome law hold everywhere, or only where cell-type identity
organises the wiring?

Piazza et al. fit lognormals to node strength S and degree k pooled over a whole connectome.
Our SBM results say the fly brain is not one regime: with a lognormal weight model the blocks
track CELL-TYPE IDENTITY, while with a geometric weight model they track RETINOTOPIC POSITION
(and the neurons carrying that map are the columnar optic neurons with published hex
coordinates). So their global fit averages over two differently organised populations.

Test: run THEIR comparison -- lognormal against the alternatives -- separately in each regime.

  MAP        columnar optic neurons with published (p, q)      -- organised by position
  OPTIC-rest optic neurons without a column assignment         -- controls for "optic vs not"
  NON-OPTIC  everything else                                   -- organised by identity

Every distribution is fitted by MLE and scored by KS, exactly as they do. KS is comparable
across the three groups only at equal n, so all groups are subsampled to the smallest, and the
whole thing is repeated over draws.

Run from src/:  python t5_regime_split.py
"""
import numpy as np
import pandas as pd
from scipy import stats
import config
import data

SCRATCH = str(config.SCRATCH_DIR)
N_DRAW = 25
RNG = np.random.default_rng(0)

# their candidates (lognormal, exponential, power law, Poisson-like) plus the two they concede
# also fit well (Weibull, Gamma)
FAMILIES = {
    "lognormal":   lambda x: stats.lognorm(*stats.lognorm.fit(x, floc=0)),
    "gamma":       lambda x: stats.gamma(*stats.gamma.fit(x, floc=0)),
    "weibull":     lambda x: stats.weibull_min(*stats.weibull_min.fit(x, floc=0)),
    "exponential": lambda x: stats.expon(*stats.expon.fit(x, floc=0)),
    "powerlaw":    lambda x: stats.pareto(*stats.pareto.fit(x, floc=0)),
    # the two DISCRETE families that matter for us: geometric is the weight model that recovers
    # retinotopy in the SBM, Poisson is one of the canonical models Piazza et al. reject.
    # NOTE: S and k are integer counts, so these are legitimate, but a KS statistic against a
    # STEP cdf is not on quite the same footing as one against a smooth cdf -- a discrete family
    # pays for its jumps. Read discrete-vs-continuous gaps with that caveat.
    "geometric":   lambda x: stats.geom(min(max(1.0 / max(x.mean(), 1.0), 1e-9), 1 - 1e-9)),
    "poisson":     lambda x: stats.poisson(max(x.mean(), 1e-9)),
}


def ks_all(x):
    out = {}
    for name, fit in FAMILIES.items():
        try:
            d = fit(x)
            out[name] = stats.kstest(x, d.cdf).statistic
        except Exception:
            out[name] = np.nan
    return out


def main():
    lab = data.load_labels()
    c = pd.read_parquet(f"{SCRATCH}/conn.parquet")
    ends = pd.concat([c.rename(columns={"pre_root_id": "id", "post_root_id": "other"}),
                      c.rename(columns={"post_root_id": "id", "pre_root_id": "other"})])
    S = ends.groupby("id").syn_count.sum().astype(float)
    k = ends.groupby("id").other.nunique().astype(float)

    hexids = set(pd.read_csv(f"{SCRATCH}/column_assignment.csv.gz").root_id.tolist())
    df = pd.DataFrame({"S": S, "k": k}).join(lab[["super_class", "primary_type"]])
    df = df[(df.S > 0) & (df.k > 0)]
    df["regime"] = np.where(df.index.isin(hexids), "MAP (columnar, has p,q)",
                     np.where(df.super_class == "optic", "OPTIC-rest", "NON-OPTIC"))
    print(df.regime.value_counts().to_string(), "\n")

    n_match = int(df.regime.value_counts().min())
    print(f"subsampling every regime to n = {n_match:,}, {N_DRAW} draws\n")

    rows = []
    for q in ["S", "k"]:
        for reg, g in df.groupby("regime"):
            v = g[q].values
            for _ in range(N_DRAW):
                x = RNG.choice(v, n_match, replace=False)
                rows.append(dict(quantity=q, regime=reg, **ks_all(x)))
    r = pd.DataFrame(rows)
    r.to_csv(f"{SCRATCH}/t5_regime.csv", index=False)

    fam = list(FAMILIES)
    print("=" * 96)
    print("KS statistic (lower = better fit), mean over draws at matched n")
    print("=" * 96)
    for q in ["S", "k"]:
        t = r[r.quantity == q].groupby("regime")[fam].mean()
        t["WINNER"] = t.idxmin(axis=1)
        t["lognormal_rank"] = t[fam].rank(axis=1).lognormal.astype(int)
        print(f"\n--- {'node strength S' if q=='S' else 'degree k'} ---")
        print(t.round(4).to_string())


if __name__ == "__main__":
    main()
