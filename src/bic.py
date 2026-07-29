"""
BIC across ALL weight families — the one label-free criterion that IS comparable four-way.

WHY THIS EXISTS
graph-tool's `entropy()` (our MDL column) prices each weight family with its own internal
priors and in its own units: discrete families encode PROBABILITIES, continuous families encode
DENSITIES. So MDL is only comparable within a family (and only on the same graph). BIC fixes
that by pricing every model ourselves under one fixed recipe:

    BIC = -2 * loglik + k * log(n)

  loglik : each edge weight scored by its OWN block-pair's fitted distribution, with every
           family put on the SAME integer support -- continuous densities are converted to
           probabilities on the unit bin around each integer, P(W=k)=F(k+.5)-F(k-.5), and all
           families are conditioned on W>=threshold (weights below it are unobservable).
           Without this, densities and probabilities are different units and the comparison is
           meaningless.
  k      : parameters, following the weighted-SBM / SBM-BIC literature (Aicher, Jacobs &
           Clauset 2014; SBM BIC uses d = K^2 + K - 1):
                k = (occupied block-pairs) * (params per family)   [the distributions]
                  + (K - 1)                                        [the partition]
  n      : number of edges scored.

Lower BIC is better. BIC/2 is the classic two-part MDL approximation (-loglik + (k/2)log n),
so this is an MDL in the textbook sense -- just with a generic parameter penalty instead of
graph-tool's family-specific priors, which is exactly what buys comparability.

CAVEATS (report them):
  - BIC assumes regular models with independent observations; here the partition is latent and
    edges in a bundle share parameters, so it is an approximation, not an exact codelength.
  - Each fit brings its OWN partition, so a BIC gap reflects the whole model (weights + its
    partition/resolution), not the weight likelihood in isolation.
  - Applying it ACROSS weight families is our extension; the cited works use it within one.

Run from src/:  python bic.py [--out ...] [--jobs N] [--refresh]
"""
import argparse
import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy import stats
import config
import data

RESULTS = config.WORK_DIR / "results"
CACHE = config.WORK_DIR / "bic_scores.csv"
RUN_RE = re.compile(r"(?P<model>[a-z]+)_t(?P<thr>\d+)_(?P<dir>dir|und)_(?P<dc>dc|ndc)_s(?P<seed>\d+)$")

# params per block-pair for each family (what BIC charges per bundle)
N_PARAMS = {"lognormal": 2, "gaussian": 2, "poisson": 1, "geometric": 1, "exponential": 1}
MIN_BUNDLE = 5          # bundles smaller than this fall back to the global fit (as in xval.py)


# ---------------- truncated MLE per bundle (consistent with truncated scoring) -------------
def fit_params(model, w, thr):
    x = np.asarray(w, float)
    if model == "geometric":
        return {"p": float(min(max(1.0 / max(x.mean() - thr + 1.0, 1.0), 1e-6), 1 - 1e-6))}
    if model == "poisson":
        from scipy.optimize import minimize_scalar
        f = lambda lam: -(stats.poisson.logpmf(x, max(lam, 1e-9)).sum()
                          - len(x) * stats.poisson.logsf(thr - 1, max(lam, 1e-9)))
        r = minimize_scalar(f, bounds=(1e-3, max(x.max(), 3 * x.mean()) + 1.0), method="bounded")
        return {"lam": float(max(r.x, 1e-9))}
    if model == "exponential":
        return {"scale": float(max(x.mean() - (thr - 1), 1e-6))}      # shifted-exponential MLE
    y = np.log(x) if model == "lognormal" else x
    return {"mu": float(y.mean()), "sigma": float(max(y.std(ddof=0), 1e-6))}


def _logdiff(a, b):
    return a + np.log(-np.expm1(np.minimum(b - a, -1e-12)))


def log_pmf(model, k, p, thr):
    """log P(W=k | W>=thr) on the COMMON integer support, stable in both tails."""
    k = np.asarray(k, float)
    if model in ("lognormal", "gaussian"):
        mu, s = p["mu"], p["sigma"]
        if model == "lognormal":
            lo, hi, mid = np.log(k - 0.5), np.log(k + 0.5), np.log(k)
            c = np.log(thr - 0.5)
        else:
            lo, hi, mid, c = k - 0.5, k + 0.5, k, thr - 0.5
        zl, zh, zm = (lo - mu) / s, (hi - mu) / s, (mid - mu) / s
        out = np.where(zm <= 0,
                       _logdiff(stats.norm.logcdf(zh), stats.norm.logcdf(zl)),
                       _logdiff(stats.norm.logsf(zl), stats.norm.logsf(zh)))
        return out - stats.norm.logsf((c - mu) / s)
    if model == "exponential":
        sc = p["scale"]
        lo, hi = np.maximum(k - 0.5, 0.0), k + 0.5
        out = _logdiff(stats.expon.logsf(lo, scale=sc), stats.expon.logsf(hi, scale=sc))
        return out - stats.expon.logsf(max(thr - 0.5, 0.0), scale=sc)
    if model == "poisson":
        return stats.poisson.logpmf(k, p["lam"]) - stats.poisson.logsf(thr - 1, p["lam"])
    if model == "geometric":
        return stats.geom.logpmf(np.maximum(k, 1), p["p"]) - stats.geom.logsf(thr - 1, p["p"])
    raise ValueError(model)


def score_fit(run_dir, thr=5):
    m = RUN_RE.match(os.path.basename(run_dir))
    model, directed = m["model"], m["dir"] == "dir"
    if model not in N_PARAMS:
        return None
    pre, post, w = data.load_edges(threshold=thr, directed=directed)
    d = np.load(os.path.join(run_dir, "partition.npz"))
    blk = pd.Series(d["blocks"], index=d["node_ids"])
    a = blk.reindex(pre).to_numpy(); b = blk.reindex(post).to_numpy()
    ok = ~(pd.isna(a) | pd.isna(b))
    a, b, wv = a[ok].astype(np.int64), b[ok].astype(np.int64), w[ok].astype(float)

    K = int(len(np.unique(np.concatenate([a, b]))))
    ua, ia = np.unique(a, return_inverse=True); ub, ib = np.unique(b, return_inverse=True)
    key = ia.astype(np.int64) * len(ub) + ib
    order = np.argsort(key, kind="stable")
    key_s, w_s = key[order], wv[order]
    bounds = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1], True])

    # partition codelength: transmitting which of K blocks each node belongs to costs
    # N * H(pi) nats, H = entropy of the block-size distribution. This is a CODELENGTH, not a
    # tunable penalty -- it is what makes the criterion charge honestly for resolution.
    _, cnts = np.unique(np.concatenate([a, b]), return_counts=True)
    pi = cnts / cnts.sum()
    H = float(-(pi * np.log(pi)).sum())
    N_nodes = int(len(blk))
    partition_nats = N_nodes * H

    glob_p = fit_params(model, wv, thr)
    ll = np.empty(len(w_s)); n_bundles = 0
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        seg = w_s[lo:hi]
        if len(seg) >= MIN_BUNDLE:
            ll[lo:hi] = log_pmf(model, seg, fit_params(model, seg, thr), thr); n_bundles += 1
        else:
            ll[lo:hi] = log_pmf(model, seg, glob_p, thr)
    loglik = float(np.sum(ll))
    n = len(w_s)
    k_dist = n_bundles * N_PARAMS[model]               # distribution parameters only
    # two-part description length (nats): data + distribution params + partition
    dl = -loglik + 0.5 * k_dist * np.log(n) + partition_nats
    return dict(run=os.path.basename(run_dir), model=model, dir=m["dir"], dc=m["dc"],
                seed=int(m["seed"]), n_blocks=K, n_edges=n, occupied_bundles=n_bundles,
                loglik=loglik, k_dist=k_dist, partition_nats=partition_nats,
                block_entropy=H, n_nodes=N_nodes, dl=float(dl),
                nats_per_edge=float(loglik / n))


def _safe(rd):
    try:
        return score_fit(rd)
    except Exception as e:
        print(f"[warn] {os.path.basename(rd)}: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    if CACHE.exists() and not a.refresh:
        df = pd.read_csv(CACHE); print(f"[bic] loaded cache ({len(df)} fits)", flush=True)
    else:
        runs = [str(p) for p in sorted(RESULTS.glob("*_t5_*"))
                if RUN_RE.match(os.path.basename(p))
                and os.path.exists(os.path.join(p, "partition.npz"))]
        print(f"[bic] scoring {len(runs)} fits on {a.jobs} cores ...", flush=True)
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            rows = [r for r in ex.map(_safe, runs) if r]
        df = pd.DataFrame(rows); df.to_csv(CACHE, index=False)
        print(f"[bic] cached -> {CACHE}", flush=True)

    g = (df.groupby(["model", "dir", "dc"])
           .agg(DL_M=("dl", lambda x: x.mean() / 1e6), sd=("dl", lambda x: x.std() / 1e6),
                data_M=("loglik", lambda x: -x.mean() / 1e6),
                part_M=("partition_nats", lambda x: x.mean() / 1e6),
                nats_edge=("nats_per_edge", "mean"), blocks=("n_blocks", "mean"),
                k_dist=("k_dist", "mean"), n=("dl", "size"))
           .reset_index().sort_values("DL_M"))
    pd.set_option("display.width", 220)
    print("\nTwo-part DESCRIPTION LENGTH, all weight families (M nats; LOWER = better)")
    print("DL = -loglik + (k_dist/2)*log(n_edges) + N*H(partition)")
    print("common integer support, truncated at W>=5 | data_M=-loglik, part_M=partition cost\n")
    print(g.round(3).to_string(index=False))
    best = g.iloc[0]
    print(f"\nBEST overall: {best.model}.{best['dir']}.{best.dc}  DL={best.DL_M:.3f}M")
    d = g[(g["dir"] == "dir") & (g.dc == "dc")]
    print("\nwithin directed+DC (canonical):")
    print(d.round(3).to_string(index=False))
    if len(d):
        print(f"WINNER: {d.iloc[0].model}")


if __name__ == "__main__":
    main()
