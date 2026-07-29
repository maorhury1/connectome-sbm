"""
Integer probability models on a COMMON support, for the candidate-partition weighted
description length (CPWDL).

Every family answers the same question: "what probability do you assign to the recorded
integer k?" -- so continuous and discrete families become comparable.

CONDITIONING (w >= WMIN). Our weights are floored at 5 by construction (FlyWire discards
sub-threshold connections), but a fitted model does not know that and would spend probability
on impossible values 0..4. Each family wastes a DIFFERENT amount, which biases the comparison
for reasons unrelated to fit quality. Every pmf here is therefore conditioned:

    q(k | W >= wmin) = q_raw(k) / P(W >= wmin),      k >= wmin
                     = 0                              otherwise

Parameters are fitted by maximising the same conditioned log-likelihood that is used for
scoring (fitting untruncated and scoring truncated would bias each family differently).

All continuous families are discretised by unit-width bins, q(k) = F(k+1/2) - F(k-1/2), so the
result is a probability, not a density. Computations are done on the numerically stable side
(log-cdf vs log-sf) so deep-tail weights get their true value instead of an artificial floor.
"""
import numpy as np
from scipy import stats
from scipy.optimize import minimize, minimize_scalar

FAMILIES = ("lognormal", "gaussian", "poisson", "geometric", "exponential")
N_PARAMS = {"lognormal": 2, "gaussian": 2, "poisson": 1, "geometric": 1, "exponential": 1}
WMIN_DEFAULT = 5


# ---------------------------------------------------------------- raw (unconditioned) pmf ---
def _logdiff(a, b):
    """log(exp(a) - exp(b)) for a >= b, stable."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a + np.log(-np.expm1(np.minimum(b - a, -1e-300)))


def _log_bin(k, logcdf, logsf, mid):
    """log[F(k+.5) - F(k-.5)] using whichever tail is numerically stable."""
    k = np.asarray(k, float)
    lo, hi = k - 0.5, k + 0.5
    upper = mid(k) > 0
    out = np.empty(k.shape, float)
    if np.any(~upper):
        out[~upper] = _logdiff(logcdf(hi[~upper]), logcdf(lo[~upper]))
    if np.any(upper):
        out[upper] = _logdiff(logsf(lo[upper]), logsf(hi[upper]))
    return out


def raw_logpmf(family, k, p):
    """log q_raw(k): unconditioned integer probability."""
    k = np.asarray(k, float)
    if family == "poisson":
        return stats.poisson.logpmf(k, p["lam"])
    if family == "geometric":                       # ZERO-based: q(k) = p(1-p)^k
        return np.log(p["p"]) + k * np.log1p(-p["p"])
    if family == "gaussian":
        mu, s = p["mu"], p["sigma"]
        return _log_bin(k, lambda v: stats.norm.logcdf((v - mu) / s),
                           lambda v: stats.norm.logsf((v - mu) / s),
                           lambda v: (v - mu) / s)
    if family == "lognormal":
        mu, s = p["mu"], p["sigma"]
        lg = lambda v: np.log(np.maximum(v, 1e-300))
        return _log_bin(k, lambda v: stats.norm.logcdf((lg(v) - mu) / s),
                           lambda v: stats.norm.logsf((lg(v) - mu) / s),
                           lambda v: (lg(np.maximum(v, 1e-300)) - mu) / s)
    if family == "exponential":
        sc = p["scale"]
        return _log_bin(k, lambda v: stats.expon.logcdf(np.maximum(v, 0.0), scale=sc),
                           lambda v: stats.expon.logsf(np.maximum(v, 0.0), scale=sc),
                           lambda v: v - sc)
    raise ValueError(family)


def log_tail(family, p, wmin):
    """log P(W >= wmin) under the raw model -- the conditioning normaliser."""
    if family == "poisson":
        return float(stats.poisson.logsf(wmin - 1, p["lam"]))
    if family == "geometric":                       # P(K >= m) = (1-p)^m
        return float(wmin * np.log1p(-p["p"]))
    c = wmin - 0.5
    if family == "gaussian":
        return float(stats.norm.logsf((c - p["mu"]) / p["sigma"]))
    if family == "lognormal":
        return float(stats.norm.logsf((np.log(c) - p["mu"]) / p["sigma"]))
    if family == "exponential":
        return float(stats.expon.logsf(c, scale=p["scale"]))
    raise ValueError(family)


def logpmf(family, k, p, wmin=WMIN_DEFAULT):
    """log q(k | W >= wmin): the scored quantity. -inf below the floor."""
    k = np.asarray(k, float)
    out = raw_logpmf(family, k, p) - log_tail(family, p, wmin)
    return np.where(k >= wmin, out, -np.inf)


# ------------------------------------------------------------------------------- fitting ---
def _init(family, w):
    x = np.asarray(w, float)
    if family == "poisson":
        return {"lam": max(x.mean(), 1e-6)}
    if family == "geometric":
        return {"p": float(min(max(1.0 / (1.0 + x.mean()), 1e-6), 1 - 1e-6))}
    if family == "exponential":
        return {"scale": max(x.mean(), 1e-6)}
    y = np.log(x) if family == "lognormal" else x
    return {"mu": float(y.mean()), "sigma": float(max(y.std(ddof=0), 1e-3))}


def _pack(family, p):
    if family == "poisson":      return np.array([np.log(p["lam"])])
    if family == "geometric":    return np.array([np.log(p["p"] / (1 - p["p"]))])
    if family == "exponential":  return np.array([np.log(p["scale"])])
    return np.array([p["mu"], np.log(p["sigma"])])


def _unpack(family, t):
    if family == "poisson":      return {"lam": float(np.exp(np.clip(t[0], -30, 30)))}
    if family == "geometric":
        z = 1.0 / (1.0 + np.exp(-np.clip(t[0], -30, 30)))
        return {"p": float(min(max(z, 1e-9), 1 - 1e-9))}
    if family == "exponential":  return {"scale": float(np.exp(np.clip(t[0], -30, 30)))}
    return {"mu": float(t[0]), "sigma": float(np.exp(np.clip(t[1], -30, 30)))}


def fit(family, w, wmin=WMIN_DEFAULT):
    """MLE of the CONDITIONED likelihood -- the same objective used for scoring.

    Returns (params, info). info.ok is False if the optimiser failed; callers must then move
    the bundle to the pooled backoff group rather than silently accepting a moment estimate.
    """
    x = np.asarray(w, float)
    p0 = _init(family, x)
    nll = lambda t: -float(np.sum(logpmf(family, x, _unpack(family, t), wmin)))

    t0 = _pack(family, p0)
    if not np.isfinite(nll(t0)):                      # nudge off a degenerate start
        if family in ("gaussian", "lognormal"):
            p0["sigma"] = max(p0["sigma"], 0.5); t0 = _pack(family, p0)
    try:
        if len(t0) == 1:
            r = minimize_scalar(lambda z: nll(np.array([z])),
                                bracket=(t0[0] - 2, t0[0] + 2), method="brent",
                                options={"maxiter": 400})
            t, ok, nit = np.array([r.x]), bool(r.success), int(getattr(r, "nit", 0))
        else:
            r = minimize(nll, t0, method="Nelder-Mead",
                         options=dict(maxiter=2000, xatol=1e-6, fatol=1e-8))
            t, ok, nit = r.x, bool(r.success), int(r.nit)
        val = nll(t)
        if not np.isfinite(val):
            return p0, {"ok": False, "reason": "nonfinite objective", "nit": nit}
        return _unpack(family, t), {"ok": ok, "nll": float(val), "nit": nit}
    except Exception as e:
        return p0, {"ok": False, "reason": f"{type(e).__name__}: {e}", "nit": 0}


# ------------------------------------------------------------------------------ sampling ---
def rvs(family, p, n, wmin=WMIN_DEFAULT, rng=None, kmax=100000):
    """Sample from the CONDITIONED model by inverse-cdf on the integer grid."""
    rng = rng or np.random.default_rng(0)
    hi = wmin
    while hi < kmax:                                  # grow until the tail is negligible
        ks = np.arange(wmin, hi + 1)
        pk = np.exp(logpmf(family, ks, p, wmin))
        if pk.sum() > 1 - 1e-9 or hi > 10 * (wmin + 1) * 100:
            break
        hi *= 4
    ks = np.arange(wmin, hi + 1)
    pk = np.exp(logpmf(family, ks, p, wmin))
    pk = pk / pk.sum()
    return rng.choice(ks, size=n, p=pk)


def total_mass(family, p, wmin=WMIN_DEFAULT, kmax=200000):
    """Sum of q(k | W>=wmin) over the integer support -- must approach 1."""
    ks = np.arange(wmin, kmax)
    return float(np.exp(logpmf(family, ks, p, wmin)).sum())
