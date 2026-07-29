"""
Peixoto's Bayesian model selection for weighted SBMs, applied to our five weight models.

WHAT THIS IS
Peixoto (arXiv:1708.01432, "Nonparametric weighted stochastic block models") selects among
weight models by POSTERIOR ODDS of the joint evidence, not by an ad-hoc information criterion:

    Lambda = [ P(A, y(x) | {b}_1, M_y) P({b}_1) prod_ij f'(x_ij)^A_ij ]
           / [ P(A, z(x) | {b}_2, M_z) P({b}_2) prod_ij g'(x_ij)^A_ij ]        (his Eq. 109)

Everything except the derivative (Jacobian) terms is exactly what graph-tool's
`state.entropy()` returns: S = -log P(A, x, {b}), i.e. adjacency + partition prior + weight
evidence, with the model's own priors and complexity already integrated in. So a valid
comparison is a difference of CORRECTED entropies, and nothing needs to be re-derived.

He demonstrates precisely our comparison (his Sec. IV G / Table I): exponential on raw x versus
normal on ln x (= lognormal), on human brain connectivity, reporting ln Lambda ~ 4458 for the
log-normal model.

THE TWO CORRECTIONS WE MUST ADD

1) JACOBIAN (transformed weights). graph-tool scores the property map it is given. For
   "lognormal" we hand it y = ln w, so its entropy is a density in y-space. Converting to
   x-space costs the Jacobian: with y = ln x, |dy/dx| = 1/x, hence

        S_x = S_y + sum_edges log(w)

   For every untransformed model (gaussian, exponential, poisson, geometric) the term is 0.
   (Peixoto: for discrete weights "one simply omits the derivative terms".)

2) COMMON MEASUREMENT MODEL (continuous vs discrete). Synapse counts are integers with a hard
   floor at w>=5. Discrete families already yield probabilities. Continuous families yield
   densities, which become probabilities on the unit bin around each integer:

        P(W=w) = F(w+1/2) - F(w-1/2) ~= p(w) * 1     ->     -log P ~= -log p  (log 1 = 0)

   so for UNIT-spaced integers the continuous entropy is already on the probability scale to
   first order; we report the exact bin correction so its size is visible rather than assumed.
   Separately, no family is conditioned on w>=5, so each wastes mass on impossible values; the
   truncation correction  sum_edges log P_bundle(W>=5)  is computed per bundle and reported.

Both corrections are reported as SEPARATE COLUMNS: the headline comparison is Peixoto's
(entropy + Jacobian), and the reader can see exactly how much the measurement-model terms move
it, instead of trusting an approximation silently.

Run from src/:   python peixoto_evidence.py [--jobs N] [--refresh]
"""
import argparse
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
CACHE = config.WORK_DIR / "peixoto_evidence.csv"
RUN_RE = re.compile(r"(?P<model>[a-z]+)_t(?P<thr>\d+)_(?P<dir>dir|und)_(?P<dc>dc|ndc)_s(?P<seed>\d+)$")

TRANSFORMED = {"lognormal"}                      # models fitted on log(w) -> need Jacobian
CONTINUOUS = {"lognormal", "gaussian", "exponential"}
MIN_BUNDLE = 5


def fit_params(model, w, thr):
    """Plain (untruncated) MLE -- matches what graph-tool's weight model is scoring."""
    x = np.asarray(w, float)
    if model == "geometric":
        return {"p": float(min(max(1.0 / max(x.mean(), 1.0000001), 1e-6), 1 - 1e-6))}
    if model == "poisson":
        return {"lam": float(max(x.mean(), 1e-9))}
    if model == "exponential":
        return {"scale": float(max(x.mean(), 1e-6))}
    y = np.log(x) if model == "lognormal" else x
    return {"mu": float(y.mean()), "sigma": float(max(y.std(ddof=0), 1e-6))}


def _cdf(model, v, p):
    if model == "lognormal":
        return stats.norm.cdf((np.log(np.maximum(v, 1e-12)) - p["mu"]) / p["sigma"])
    if model == "gaussian":
        return stats.norm.cdf((v - p["mu"]) / p["sigma"])
    if model == "exponential":
        return stats.expon.cdf(np.maximum(v, 0.0), scale=p["scale"])
    if model == "poisson":
        return stats.poisson.cdf(np.floor(v), p["lam"])
    if model == "geometric":
        return stats.geom.cdf(np.maximum(np.floor(v), 0), p["p"])
    raise ValueError(model)


def corrections(run_dir, thr=5):
    """Jacobian + measurement-model corrections for one fit."""
    m = RUN_RE.match(os.path.basename(run_dir))
    model, directed = m["model"], m["dir"] == "dir"
    summ = json.load(open(os.path.join(run_dir, "summary.json")))
    S_gt = float(summ["mdl_entropy"])                       # graph-tool: -log P(A, x, {b})

    pre, post, w = data.load_edges(threshold=thr, directed=directed)
    d = np.load(os.path.join(run_dir, "partition.npz"))
    blk = pd.Series(d["blocks"], index=d["node_ids"])
    a = blk.reindex(pre).to_numpy(); b = blk.reindex(post).to_numpy()
    ok = ~(pd.isna(a) | pd.isna(b))
    a, b, wv = a[ok].astype(np.int64), b[ok].astype(np.int64), w[ok].astype(float)
    n = len(wv)

    # (1) Jacobian: y = ln x  ->  S_x = S_y + sum log w
    jac = float(np.log(wv).sum()) if model in TRANSFORMED else 0.0

    # (2) measurement model, per bundle: unit-bin discretization + truncation at w>=thr
    ua, ia = np.unique(a, return_inverse=True); ub, ib = np.unique(b, return_inverse=True)
    key = ia.astype(np.int64) * len(ub) + ib
    order = np.argsort(key, kind="stable")
    key_s, w_s = key[order], wv[order]
    bounds = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1], True])
    glob = fit_params(model, wv, thr)

    bin_corr = 0.0        # -log P(bin) - (-log density)  : exact minus first-order
    trunc_corr = 0.0      # + sum log P(W >= thr)         : conditioning on the observable range
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        seg = w_s[lo:hi]
        p = fit_params(model, seg, thr) if len(seg) >= MIN_BUNDLE else glob
        tail = 1.0 - _cdf(model, thr - 0.5 if model in CONTINUOUS else thr - 1, p)
        trunc_corr += len(seg) * np.log(max(tail, 1e-300))
        if model in CONTINUOUS:                              # exact bin mass vs density
            pb = np.maximum(_cdf(model, seg + 0.5, p) - _cdf(model, seg - 0.5, p), 1e-300)
            if model == "lognormal":
                dens = stats.norm.pdf((np.log(seg) - p["mu"]) / p["sigma"]) / (p["sigma"] * seg)
            elif model == "gaussian":
                dens = stats.norm.pdf((seg - p["mu"]) / p["sigma"]) / p["sigma"]
            else:
                dens = stats.expon.pdf(seg, scale=p["scale"])
            bin_corr += float(np.sum(-np.log(pb) + np.log(np.maximum(dens, 1e-300))))

    return dict(run=os.path.basename(run_dir), model=model, dir=m["dir"], dc=m["dc"],
                seed=int(m["seed"]), n_blocks=int(summ["n_blocks"]), n_edges=n,
                S_gt=S_gt, jacobian=jac, bin_corr=float(bin_corr),
                trunc_corr=float(trunc_corr),
                S_peixoto=S_gt + jac,                        # Eq. 109 headline
                S_common=S_gt + jac + float(bin_corr) + float(trunc_corr))


def _safe(rd):
    try:
        return corrections(rd)
    except Exception as e:
        print(f"[warn] {os.path.basename(rd)}: {type(e).__name__}: {str(e)[:90]}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    if CACHE.exists() and not a.refresh:
        df = pd.read_csv(CACHE); print(f"[peixoto] cache ({len(df)} fits)", flush=True)
    else:
        runs = [str(p) for p in sorted(RESULTS.glob("*_t5_*"))
                if RUN_RE.match(os.path.basename(p))
                and os.path.exists(os.path.join(p, "partition.npz"))]
        print(f"[peixoto] {len(runs)} fits on {a.jobs} cores ...", flush=True)
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            rows = [r for r in ex.map(_safe, runs) if r]
        df = pd.DataFrame(rows); df.to_csv(CACHE, index=False)
        print(f"[peixoto] cached -> {CACHE}", flush=True)

    M = 1e6
    g = (df.groupby(["model", "dir", "dc"])
           .agg(S_gt=("S_gt", lambda x: x.mean() / M), jac=("jacobian", lambda x: x.mean() / M),
                S_peix=("S_peixoto", lambda x: x.mean() / M),
                sd=("S_peixoto", lambda x: x.std() / M),
                bin_c=("bin_corr", lambda x: x.mean() / M),
                trunc=("trunc_corr", lambda x: x.mean() / M),
                S_common=("S_common", lambda x: x.mean() / M),
                blocks=("n_blocks", "mean"), n=("S_gt", "size"))
           .reset_index().sort_values("S_peix"))
    pd.set_option("display.width", 230)
    print("\nPeixoto evidence (M nats; LOWER = better). S_peix = graph-tool entropy + Jacobian")
    print("S_common additionally puts every family on integer probabilities truncated at w>=5\n")
    print(g.round(3).to_string(index=False))
    for tag, sub in (("directed + DC (canonical)", g[(g["dir"] == "dir") & (g.dc == "dc")]),):
        print(f"\n--- {tag} ---")
        print(sub.round(3).to_string(index=False))
        if len(sub) > 1:
            s = sub.sort_values("S_peix")
            w0, w1 = s.iloc[0], s.iloc[1]
            print(f"Eq.109 winner: {w0.model}; ln(Lambda) vs {w1.model} = "
                  f"{(w1.S_peix - w0.S_peix) * M:,.0f} nats")
            c = sub.sort_values("S_common")
            print(f"common-measure winner: {c.iloc[0].model}")


if __name__ == "__main__":
    main()
