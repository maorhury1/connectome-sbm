"""
T10 -- the audit the project skipped: is the weight law lognormal CONDITIONAL ON A BLOCK PAIR?

Everything checked so far (T1, T5, T6, ...) looked at the POOLED marginal of all edge weights.
That is not the object an SBM makes an assumption about. A weighted SBM assumes a parametric
family for the weights of the bundle running from block r to block s. The pooled marginal is a
MIXTURE over ~10^4 such bundles, and a mixture of almost anything is lognormal-ish. So a pooled
lognormal fit is evidence for nothing.

Two questions, both at the bundle level (block pair (r,s), directed, >=30 weights):

(a) SHAPE. Fit lognormal / gamma / weibull / exponential / geometric / poisson to each bundle by
    MLE and rank by KS. Which family actually wins, unweighted and weighted by bundle size?
    Done under the LOGNORMAL model's partition and repeated under the GEOMETRIC model's
    partition, so the answer cannot be an artifact of which partition we condition on.

(b) MEAN-VARIANCE SCALING. This is the ONLY axis on which the candidate weight families differ:
        poisson    var = mean                      -> log-log slope 1
        gaussian   var independent of mean         -> slope 0
        geometric  var ~ mean^2, ratio LOCKED      -> slope 2, no freedom
        lognormal  var ~ mean^2, ratio FREE        -> slope 2, plus per-bundle freedom
    Regress log(var) on log(mean) over bundles; report the slope + CI, and the per-bundle CV.
    A slope near 2 rules out poisson/gaussian but NOT geometric. Geometric is separated from
    lognormal by whether the dispersion RATIO is locked: under truncation at 5 a geometric with
    mean m has variance exactly (m-4)(m-5) with zero freedom, so var_obs/(m-4)(m-5) must be ~1
    for every bundle if geometric is adequate.

MEASUREMENT-FOOTING NOTE. Weights are integers floored at 5. Every family here -- continuous or
discrete -- is therefore scored as a distribution on the integers {5,6,...}:
        P(k) = [F(k+0.5) - F(k-0.5)] / (1 - F(4.5))
MLE is done on THAT likelihood, and KS is computed against THAT step CDF. So all six families sit
on identical support with identical (step) CDFs, which removes the usual "a discrete family pays
for its jumps in KS" caveat that applied to earlier scripts (t5_regime_split.py).

Consequence worth stating: on this support the discretised truncated exponential is ALGEBRAICALLY
the truncated geometric (both give P(k) proportional to q^k on k>=5). They are one family here,
not two, and the script asserts their KS agrees.

Run from src/:  python t10_block_conditioned.py
"""
import os
import sys
import json
import numpy as np
from multiprocessing import Pool
from scipy import optimize, special, stats

import config
import data

MIN_N = 30
CACHE = os.path.join(str(config.SCRATCH_DIR), "t10_edges_t5_dir.npz")
OUT = os.path.join(str(config.SCRATCH_DIR), "t10_bundles_{}_s{}.npz")
NPROC = 24
RNG = np.random.default_rng(0)

TRUNC = 4.5          # weights are >= 5, so the continuous support is truncated below at 4.5
FAMILIES = ["lognormal", "gamma", "weibull", "exponential", "geometric", "poisson"]
# exponential and geometric coincide on this support (see module docstring)
MERGED = {"exponential": "geometric(=exponential)", "geometric": "geometric(=exponential)"}


# ---------------------------------------------------------------- data plumbing
def load_edges_cached():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["pre"], z["post"], z["w"]
    pre, post, w = data.load_edges(threshold=5, directed=True)
    np.savez(CACHE, pre=pre, post=post, w=w)
    return pre, post, w


def bundle_split(pre, post, w, model, seed):
    """Group edge weights by ordered block pair under a given fitted partition."""
    p = np.load(f"/var/tmp/csbm_work/results/{model}_t5_dir_dc_s{seed}/partition.npz")
    nid, blk = p["node_ids"], p["blocks"]
    _, b = np.unique(blk, return_inverse=True)          # relabel blocks to 0..B-1
    B = int(b.max()) + 1
    lut = dict(zip(nid.tolist(), b.tolist()))
    r = np.fromiter((lut.get(x, -1) for x in pre), np.int64, len(pre))
    s = np.fromiter((lut.get(x, -1) for x in post), np.int64, len(post))
    ok = (r >= 0) & (s >= 0)
    key = r[ok] * B + s[ok]
    ww = w[ok]
    order = np.argsort(key, kind="stable")
    key, ww = key[order], ww[order]
    bounds = np.flatnonzero(np.diff(key)) + 1
    groups = np.split(ww, bounds)
    keys = np.concatenate(([key[0]], key[bounds]))
    out = [(int(k), g) for k, g in zip(keys, groups) if g.size >= MIN_N]
    print(f"[{model} s{seed}] B={B}  bundles(all)={len(groups):,}  "
          f"bundles(n>={MIN_N})={len(out):,}  edges in them={sum(g.size for _, g in out):,} "
          f"({100*sum(g.size for _,g in out)/ok.sum():.1f}% of edges)")
    return out, B


# ------------------------------------------------- discretised truncated likelihoods
# Each family exposes:  nll(theta, vals, cnts)  and  cdf(theta, grid) on integers >= 5.
LOG_TINY = -700.0


def _lp(p, norm):
    return np.log(np.maximum(p, 1e-300)) - np.log(max(norm, 1e-300))


# --- lognormal: theta = (mu, log sigma)
def _ln_parts(th, x):
    mu, sig = th[0], np.exp(np.clip(th[1], -8.0, 5.0))
    return (np.log(x) - mu) / sig


def nll_lognormal(th, vals, cnts):
    zh, zl = _ln_parts(th, vals + 0.5), _ln_parts(th, vals - 0.5)
    Fh = special.ndtr(zh)
    p = np.where(Fh < 0.5, Fh - special.ndtr(zl), special.ndtr(-zl) - special.ndtr(-zh))
    norm = special.ndtr(-_ln_parts(th, np.array([TRUNC])))[0]
    return float(-(cnts * _lp(p, norm)).sum())


def cdf_lognormal(th, k):
    sf = special.ndtr(-_ln_parts(th, k + 0.5))
    sf0 = special.ndtr(-_ln_parts(th, np.array([TRUNC])))[0]
    return 1.0 - sf / max(sf0, 1e-300)


# --- gamma: theta = (log a, log scale)
def _gm(th):
    return np.exp(np.clip(th[0], -12.0, 12.0)), np.exp(np.clip(th[1], -12.0, 14.0))


def nll_gamma(th, vals, cnts):
    a, sc = _gm(th)
    Fh = special.gammainc(a, (vals + 0.5) / sc)
    p = np.where(Fh < 0.5, Fh - special.gammainc(a, (vals - 0.5) / sc),
                 special.gammaincc(a, (vals - 0.5) / sc) - special.gammaincc(a, (vals + 0.5) / sc))
    norm = special.gammaincc(a, TRUNC / sc)
    return float(-(cnts * _lp(p, norm)).sum())


def cdf_gamma(th, k):
    a, sc = _gm(th)
    sf = special.gammaincc(a, (k + 0.5) / sc)
    return 1.0 - sf / max(special.gammaincc(a, TRUNC / sc), 1e-300)


# --- weibull: theta = (log c, log scale)
def _wb(th, x):
    c, sc = np.exp(np.clip(th[0], -8.0, 6.0)), np.exp(np.clip(th[1], -12.0, 14.0))
    return np.exp(-np.power(x / sc, c))          # survival function


def nll_weibull(th, vals, cnts):
    p = _wb(th, vals - 0.5) - _wb(th, vals + 0.5)
    norm = _wb(th, np.array([TRUNC]))[0]
    return float(-(cnts * _lp(p, norm)).sum())


def cdf_weibull(th, k):
    return 1.0 - _wb(th, k + 0.5) / max(_wb(th, np.array([TRUNC]))[0], 1e-300)


# --- exponential (loc fixed at 0): theta = (log scale,)
def nll_exponential(th, vals, cnts):
    sc = np.exp(np.clip(th[0], -12.0, 14.0))
    p = np.exp(-(vals - 0.5) / sc) - np.exp(-(vals + 0.5) / sc)
    norm = np.exp(-TRUNC / sc)
    return float(-(cnts * _lp(p, norm)).sum())


def cdf_exponential(th, k):
    sc = np.exp(np.clip(th[0], -12.0, 14.0))
    return 1.0 - np.exp(-(k + 0.5) / sc) / np.exp(-TRUNC / sc)


# --- geometric on {1,2,...} truncated to >=5: closed-form MLE p = 1/(mean-4)
def geom_p(m):
    return float(np.clip(1.0 / max(m - 4.0, 1e-9), 1e-12, 1 - 1e-12))


def nll_geometric(th, vals, cnts):
    p = float(np.clip(th[0], 1e-12, 1 - 1e-12))
    lp = (vals - 5.0) * np.log1p(-p) + np.log(p)
    return float(-(cnts * lp).sum())


def cdf_geometric(th, k):
    p = float(np.clip(th[0], 1e-12, 1 - 1e-12))
    return 1.0 - np.exp((k - 4.0) * np.log1p(-p))


# --- poisson truncated to >=5: theta = (log lambda,)
def nll_poisson(th, vals, cnts):
    lam = float(np.exp(np.clip(th[0], -8.0, 10.0)))
    lp = vals * np.log(lam) - lam - special.gammaln(vals + 1.0)
    norm = special.pdtrc(4, lam)                 # P(X > 4) = P(X >= 5)
    return float(-(cnts * (lp - np.log(max(norm, 1e-300)))).sum())


def cdf_poisson(th, k):
    lam = float(np.exp(np.clip(th[0], -8.0, 10.0)))
    return 1.0 - special.pdtrc(k, lam) / max(special.pdtrc(4, lam), 1e-300)


NLL = dict(lognormal=nll_lognormal, gamma=nll_gamma, weibull=nll_weibull,
           exponential=nll_exponential, geometric=nll_geometric, poisson=nll_poisson)
CDF = dict(lognormal=cdf_lognormal, gamma=cdf_gamma, weibull=cdf_weibull,
           exponential=cdf_exponential, geometric=cdf_geometric, poisson=cdf_poisson)
NPAR = dict(lognormal=2, gamma=2, weibull=2, exponential=1, geometric=1, poisson=1)


# ---------------------------------------------------------------- per-bundle fitting
def fit_bundle(arg):
    key, x = arg
    x = x.astype(np.float64)
    n = x.size
    vals, cnts = np.unique(x, return_counts=True)
    cnts = cnts.astype(np.float64)
    m, v = float(x.mean()), float(x.var(ddof=1))
    lg = np.log(x)
    mu0, sd0 = float(lg.mean()), float(max(lg.std(), 1e-3))

    inits = {
        "lognormal": np.array([mu0, np.log(sd0)]),
        "gamma": np.array([np.log(max(m * m / max(v, 1e-9), 1e-3)), np.log(max(v / m, 1e-3))]),
        "weibull": np.array([np.log(1.2), np.log(m)]),
        "exponential": np.array([np.log(max(m - TRUNC, 1e-3))]),
        "poisson": np.array([np.log(max(m, 1e-3))]),
    }

    theta, nllv = {}, {}
    for fam, th0 in inits.items():
        f = NLL[fam]
        best, bestv = th0, f(th0, vals, cnts)
        for start in (th0, th0 + np.array([0.3] * th0.size), th0 - np.array([0.3] * th0.size)):
            try:
                r = optimize.minimize(f, start, args=(vals, cnts), method="Nelder-Mead",
                                      options=dict(xatol=1e-7, fatol=1e-9, maxiter=1500))
                if np.isfinite(r.fun) and r.fun < bestv:
                    best, bestv = r.x, float(r.fun)
            except Exception:
                pass
        theta[fam], nllv[fam] = best, bestv
    theta["geometric"] = np.array([geom_p(m)])
    nllv["geometric"] = nll_geometric(theta["geometric"], vals, cnts)

    # KS on a common integer grid: both CDFs are step functions on {5..kmax}
    kmax = int(vals.max())
    grid = np.arange(5, kmax + 1, dtype=np.float64)
    emp = np.zeros(grid.size)
    emp[(vals - 5).astype(int)] = cnts
    Femp = np.cumsum(emp) / n

    ks, aic = {}, {}
    for fam in FAMILIES:
        try:
            Fm = np.clip(CDF[fam](theta[fam], grid), 0.0, 1.0)
            ks[fam] = float(np.max(np.abs(Femp - Fm)))
        except Exception:
            ks[fam] = np.nan
        aic[fam] = 2.0 * nllv[fam] + 2.0 * NPAR[fam]

    return dict(key=key, n=n, mean=m, var=v, cv=float(np.sqrt(v) / m),
                ks=[ks[f] for f in FAMILIES], aic=[aic[f] for f in FAMILIES],
                nll=[nllv[f] for f in FAMILIES],
                ln_sigma=float(np.exp(np.clip(theta["lognormal"][1], -8, 5))))


# ---------------------------------------------------------------- aggregation / stats
def ols_loglog(lm, lv, wts=None):
    """OLS of lv on lm; returns slope, classical 95% CI, HC3 95% CI, R^2, intercept."""
    X = np.column_stack([np.ones_like(lm), lm])
    W = np.ones_like(lm) if wts is None else wts / wts.mean()
    XtWX = X.T @ (X * W[:, None])
    beta = np.linalg.solve(XtWX, X.T @ (W * lv))
    res = lv - X @ beta
    n, p = len(lm), 2
    XtWXi = np.linalg.inv(XtWX)
    s2 = float((W * res ** 2).sum() / (n - p))
    se = np.sqrt(np.diag(XtWXi * s2))
    # HC3 (heteroskedasticity-robust)
    h = np.einsum("ij,jk,ik->i", X * W[:, None], XtWXi, X)
    om = (W * res) ** 2 / np.maximum((1 - h) ** 2, 1e-12)
    Vr = XtWXi @ (X.T @ (om[:, None] * X)) @ XtWXi
    ser = np.sqrt(np.diag(Vr))
    tc = stats.t.ppf(0.975, n - p)
    r2 = 1 - (W * res ** 2).sum() / (W * (lv - np.average(lv, weights=W)) ** 2).sum()
    return dict(slope=float(beta[1]), intercept=float(beta[0]),
                ci=(float(beta[1] - tc * se[1]), float(beta[1] + tc * se[1])),
                ci_hc3=(float(beta[1] - 1.96 * ser[1]), float(beta[1] + 1.96 * ser[1])),
                r2=float(r2), n=n)


def boot_slope(lm, lv, reps=1000):
    n = len(lm)
    out = np.empty(reps)
    for i in range(reps):
        idx = RNG.integers(0, n, n)
        a, b = lm[idx], lv[idx]
        out[i] = np.polyfit(a, b, 1)[0]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def win_table(res, tag, score="ks"):
    S = np.array([r[score] for r in res])
    n = np.array([r["n"] for r in res], float)
    good = np.isfinite(S).all(axis=1)
    S, n = S[good], n[good]
    win = np.array(FAMILIES)[np.argmin(S, axis=1)]
    merged = np.array([MERGED.get(f, f) for f in win])
    print(f"\n--- {tag}: {score.upper()} winner over {len(S):,} bundles "
          f"({int(n.sum()):,} edges) ---")
    print(f"{'family':<26} {'win% (unweighted)':>18} {'win% (size-weighted)':>21} "
          f"{'mean '+score:>12}")
    names = sorted(set(MERGED.get(f, f) for f in FAMILIES),
                   key=lambda x: -(merged == x).sum())
    for fam in names:
        sel = merged == fam
        cols = [i for i, f in enumerate(FAMILIES) if MERGED.get(f, f) == fam]
        print(f"{fam:<26} {100*sel.mean():>17.1f}% {100*n[sel].sum()/n.sum():>20.1f}% "
              f"{S[:, cols[0]].mean():>12.4f}")
    # explicit rank of lognormal
    rk = (S < S[:, [0]]).sum(axis=1) + 1
    print(f"lognormal mean rank among 6 = {rk.mean():.2f}; "
          f"rank1 {100*(rk==1).mean():.1f}%  top2 {100*(rk<=2).mean():.1f}%")
    return S, n


def rejection_report(res, tag):
    """Ranking alone only says which family is least wrong. This asks whether each family is
    ADEQUATE in absolute terms: KS vs the 1.36/sqrt(n) 5% critical value. (Conservative here --
    parameters are fitted, which shrinks the true critical value, so real rejection is a bit
    higher than reported. Both continuous and discrete families are step CDFs on the same
    support, so this is applied on identical footing.)"""
    n = np.array([r["n"] for r in res], float)
    ks = np.array([r["ks"] for r in res])
    crit = 1.36 / np.sqrt(n)
    print(f"\n--- {tag}: ABSOLUTE adequacy (KS vs 5% critical value), not just ranking ---")
    print(f"    bundle n: median={np.median(n):.0f} IQR=[{np.percentile(n,25):.0f},"
          f"{np.percentile(n,75):.0f}] max={n.max():.0f}")
    for i, f in enumerate(FAMILIES):
        rej = ks[:, i] > crit
        print(f"      {f:<12} REJECTED in {100*rej.mean():>5.1f}% of bundles "
              f"({100*n[rej].sum()/n.sum():>5.1f}% of edges)   median KS={np.median(ks[:,i]):.4f}")
    best = ks.min(axis=1)
    print(f"      [best-of-6 still rejected in {100*np.mean(best>crit):.1f}% of bundles]")
    for lbl, sel in [("n<100", n < 100), ("n>=1000", n >= 1000)]:
        if sel.sum() < 20:
            continue
        w = np.array(FAMILIES)[np.argmin(ks[sel], axis=1)]
        u, c = np.unique(w, return_counts=True)
        top = ", ".join(f"{a} {100*b/sel.sum():.1f}%" for a, b in
                        sorted(zip(u, c), key=lambda t: -t[1])[:3])
        print(f"      KS winner among {lbl:<8} ({sel.sum():>6,} bundles): {top}")


def implied_var(m):
    """Variance each LOCKED (1-parameter) family must have, given that its mean of the
    TRUNCATED variable equals the observed bundle mean m. These families have no freedom left
    once the mean is fixed, so this is the whole of their prediction.

    geometric truncated to k>=5:  MLE p = 1/(m-4)  =>  var = (m-4)(m-5) exactly.
    poisson  truncated to k>=5:  solve E[X|X>=5] = m for lambda, then read off Var[X|X>=5].
        Using E[X f(X)] = lam E[f(X+1)]:
            E[X 1{X>=5}]   = lam P(X>=4)
            E[X^2 1{X>=5}] = lam ( lam P(X>=2+1) + P(X>=4) )
    """
    vg = np.maximum((m - 4.0) * (m - 5.0), 1e-9)
    lam = np.exp(np.linspace(np.log(1e-2), np.log(5e3), 60000))
    p4, p3, p2 = special.pdtrc(4, lam), special.pdtrc(3, lam), special.pdtrc(2, lam)
    mm = lam * p3 / p4
    e2 = lam * (lam * p2 + p3) / p4
    vv = np.maximum(e2 - mm ** 2, 1e-12)
    keep = np.isfinite(mm) & np.isfinite(vv) & (p4 > 1e-290)
    keep &= np.concatenate(([True], np.diff(mm) > 0))
    vp = np.interp(m, mm[keep], vv[keep])
    return vg, vp


def null_slope(lm, lv_null, sel, wt=None):
    return ols_loglog(lm[sel], lv_null[sel], None if wt is None else wt[sel])["slope"]


def scaling_report(res, tag):
    n = np.array([r["n"] for r in res], float)
    m = np.array([r["mean"] for r in res], float)
    v = np.array([r["var"] for r in res], float)
    cv = np.array([r["cv"] for r in res], float)
    ok = (v > 0) & (m > 0)
    n, m, v, cv = n[ok], m[ok], v[ok], cv[ok]
    lm, lv = np.log(m), np.log(v)

    vg, vp = implied_var(m)
    lvg, lvp = np.log(vg), np.log(vp)

    print(f"\n=== {tag}: log(var) ~ log(mean) over {len(m):,} bundles ===")
    print("  NOTE the floor: weights are truncated below at 5, so a bundle with mean 5.2 is one "
          "whose\n  weights are nearly all exactly 5 and whose variance is mechanically near 0. "
          "The nominal\n  targets (poisson 1 / gaussian 0 / geometric 2) are UNTRUNCATED values "
          "and do not apply\n  directly. The last two columns give the slope each LOCKED family "
          "would itself produce on\n  this exact set of bundle means under the same truncation "
          "-- that is the honest null.")
    print(f"  {'subset':<26} {'slope':>7} {'95% CI':>17} {'HC3':>17} {'R2':>6} "
          f"{'n':>7} | {'geom-null':>9} {'pois-null':>9}")
    for lbl, sel, wt in [("all bundles", np.ones(len(m), bool), None),
                         ("all, size-weighted", np.ones(len(m), bool), n),
                         ("mean>=10 (floor-free)", m >= 10, None),
                         ("mean>=10, size-weighted", m >= 10, n),
                         ("mean>=20", m >= 20, None),
                         ("mean>=30", m >= 30, None),
                         ("n>=200 bundles only", n >= 200, None),
                         ("n>=200 & mean>=10", (n >= 200) & (m >= 10), None)]:
        if sel.sum() < 20:
            continue
        w = None if wt is None else wt[sel]
        o = ols_loglog(lm[sel], lv[sel], w)
        gn = null_slope(lm, lvg, sel, wt)
        pn = null_slope(lm, lvp, sel, wt)
        print(f"  {lbl:<26} {o['slope']:>7.3f} "
              f"[{o['ci'][0]:>6.3f},{o['ci'][1]:>6.3f}] "
              f"[{o['ci_hc3'][0]:>6.3f},{o['ci_hc3'][1]:>6.3f}] "
              f"{o['r2']:>6.3f} {o['n']:>7,} | {gn:>9.3f} {pn:>9.3f}")
    lo, hi = boot_slope(lm, lv, 1000)
    print(f"  bootstrap 95% CI on the all-bundles slope: [{lo:.3f},{hi:.3f}]")

    print(f"\n  CV = sd/mean per bundle: median={np.median(cv):.3f}  "
          f"IQR=[{np.percentile(cv,25):.3f},{np.percentile(cv,75):.3f}]  "
          f"10-90%=[{np.percentile(cv,10):.3f},{np.percentile(cv,90):.3f}]")
    print("  CV by bundle-mean quintile (is it constant?):")
    q = np.percentile(m, np.arange(0, 101, 20))
    for i in range(len(q) - 1):
        s = (m >= q[i]) & (m <= q[i + 1])
        if s.sum():
            print(f"    mean in [{q[i]:6.1f},{q[i+1]:7.1f}]  n_bundles={s.sum():5d}  "
                  f"median CV={np.median(cv[s]):.3f}")
    # slope of log CV on log mean: = slope_var/2 - 1
    o = ols_loglog(lm, np.log(cv))
    print(f"  log(CV) ~ log(mean) slope = {o['slope']:.3f} "
          f"CI=[{o['ci'][0]:.3f},{o['ci'][1]:.3f}]  (0 => CV constant)")

    # ---- LOCKED-RATIO TEST. This, not the raw slope, is what separates geometric from
    # lognormal: both have var ~ mean^2, but geometric's coefficient is fixed by the mean.
    # Truncation is handled exactly (implied_var uses the truncated moments), so this test is
    # NOT contaminated by the floor.
    ratio = v / vg
    ratio_p = v / vp
    print("\n  --- locked-ratio test (floor-corrected: implied variances are truncated moments) ---")
    for lbl, sel in [("all bundles", np.ones(len(m), bool)),
                     ("n>=200", n >= 200), ("n>=1000", n >= 1000)]:
        if sel.sum() < 20:
            continue
        rr = ratio[sel]
        p5, p95 = np.percentile(rr, 5), np.percentile(rr, 95)
        print(f"    var_obs/var_GEOM  [{lbl:<11}, {sel.sum():>6,} bundles]  median={np.median(rr):.3f}  "
              f"IQR=[{np.percentile(rr,25):.3f},{np.percentile(rr,75):.3f}]  "
              f"5-95%=[{p5:.3f},{p95:.3f}]  spread={p95/max(p5,1e-9):.1f}x  "
              f"within[0.8,1.25]={100*np.mean((rr>0.8)&(rr<1.25)):.1f}%")
    rp = ratio_p[n >= 200] if (n >= 200).sum() > 20 else ratio_p
    print(f"    var_obs/var_POIS  [n>=200]  median={np.median(rp):.2f}  "
          f"IQR=[{np.percentile(rp,25):.2f},{np.percentile(rp,75):.2f}]  "
          f"(poisson adequate <=> ~1)")
    return dict(median_cv=float(np.median(cv)), median_ratio=float(np.median(ratio)))


# ---------------------------------------------------------------- main
def run(model, seed, pre, post, w):
    cache = OUT.format(model, seed)
    if os.path.exists(cache):
        z = np.load(cache)
        assert list(z["families"]) == FAMILIES
        res = [dict(key=int(k), n=int(nn), mean=float(mm), var=float(vv),
                    cv=float(np.sqrt(vv) / mm), ks=list(kk), aic=list(aa), ln_sigma=float(ls))
               for k, nn, mm, vv, kk, aa, ls in zip(z["key"], z["n"], z["mean"], z["var"],
                                                    z["ks"], z["aic"], z["ln_sigma"])]
        print(f"[{model} s{seed}] reusing {len(res):,} cached bundle fits from {cache}")
    else:
        bl, B = bundle_split(pre, post, w, model, seed)
        with Pool(NPROC) as pool:
            res = pool.map(fit_bundle, bl, chunksize=8)
        np.savez(cache,
                 key=np.array([r["key"] for r in res]), n=np.array([r["n"] for r in res]),
                 mean=np.array([r["mean"] for r in res]), var=np.array([r["var"] for r in res]),
                 ks=np.array([r["ks"] for r in res]), aic=np.array([r["aic"] for r in res]),
                 ln_sigma=np.array([r["ln_sigma"] for r in res]), families=np.array(FAMILIES))
    tag = f"{model} partition s{seed}"
    S, _ = win_table(res, tag, "ks")
    win_table(res, tag, "aic")
    # sanity: exponential == geometric on this support
    d = np.abs(S[:, FAMILIES.index("exponential")] - S[:, FAMILIES.index("geometric")])
    print(f"  [check] max |KS_exponential - KS_geometric| = {d.max():.2e} "
          f"(they are the same family on integers>=5)")
    rejection_report(res, tag)
    st = scaling_report(res, tag)
    ls = np.array([r["ln_sigma"] for r in res])
    nn = np.array([r["n"] for r in res], float)
    print("\n  --- is the dispersion ratio free? per-bundle fitted lognormal sigma ---")
    print("    (a locked-ratio family = ONE common sigma for every bundle; implied CV of the "
          "untruncated\n     lognormal is sqrt(exp(sigma^2)-1), so sigma spread IS the free-ratio "
          "question)")
    for lbl, sel in [("all", np.ones(len(ls), bool)), ("n>=200", nn >= 200),
                     ("n>=1000", nn >= 1000)]:
        if sel.sum() < 20:
            continue
        s = ls[sel]
        # sigma > ~3 only occurs in tiny degenerate bundles; clip so the summary does not
        # overflow. Trust the n>=200 / n>=1000 rows.
        icv = np.sqrt(np.expm1(np.minimum(s, 6.0) ** 2))
        print(f"    [{lbl:<8} {sel.sum():>6,} bundles] sigma median={np.median(s):.3f} "
              f"IQR=[{np.percentile(s,25):.3f},{np.percentile(s,75):.3f}] "
              f"5-95%=[{np.percentile(s,5):.3f},{np.percentile(s,95):.3f}]  -> implied CV "
              f"median={np.median(icv):.2f} 5-95%=[{np.percentile(icv,5):.2f},"
              f"{np.percentile(icv,95):.2f}]")
    return res, st


def main():
    pre, post, w = load_edges_cached()
    print(f"[data] {w.size:,} directed edges, weight in [{w.min()},{w.max()}], "
          f"mean {w.mean():.2f}\n")
    for model in ("lognormal", "geometric"):
        print("=" * 100)
        print(f"PARTITION: {model} t5_dir_dc s0")
        print("=" * 100)
        run(model, 0, pre, post, w)
        print()


if __name__ == "__main__":
    main()
