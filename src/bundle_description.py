"""Sparse block-pair bundles, small-bundle pooling, and the BIC parameter cost (CPWDL S_W,S_th).

Bundle = the multiset of weights on edges whose ordered endpoint blocks are (r,s).

SMALL-BUNDLE RULE (spec Sec. 7): 2-parameter families are not identifiable from tiny or
constant bundles, and BIC is a large-sample approximation. A bundle gets its own parameters
only if it has >= min_size weights AND >= min_unique distinct values; all remaining bundles for
that (partition, family) are POOLED into a single shared backoff group, fitted only from the
weights it scores and charged once. The same grouping is applied to every family so the
comparison stays structurally fair.
"""
import numpy as np
import integer_weight_models as IWM


def make_bundles(src_blocks, dst_blocks, weights):
    """-> (starts, ends, order) into a weight array sorted by ordered block pair."""
    src_blocks = np.asarray(src_blocks, np.int64)
    dst_blocks = np.asarray(dst_blocks, np.int64)
    ub, ib = np.unique(dst_blocks, return_inverse=True)
    _, ia = np.unique(src_blocks, return_inverse=True)
    key = ia.astype(np.int64) * len(ub) + ib
    order = np.argsort(key, kind="stable")
    ks = key[order]
    edges = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    return edges[:-1], edges[1:], order


def group_bundles(w_sorted, starts, ends, min_size=20, min_unique=3):
    """Split bundles into locally-fitted groups and one pooled backoff group."""
    local, pooled = [], []
    for lo, hi in zip(starts, ends):
        seg = w_sorted[lo:hi]
        if len(seg) >= min_size and len(np.unique(seg)) >= min_unique:
            local.append(seg)
        else:
            pooled.append(seg)
    pooled = np.concatenate(pooled) if pooled else np.empty(0)
    return local, pooled


def score_groups(family, local, pooled, wmin=IWM.WMIN_DEFAULT):
    """-> dict with weight NLL, parameter DL, and optimiser diagnostics.

    A group whose optimiser FAILS is demoted into the pooled group (spec Sec. 8.2): no silent
    moment-estimate fallback is used in the primary result.
    """
    d = IWM.N_PARAMS[family]
    nll = 0.0
    par_dl = 0.0
    n_local = 0
    failures = 0
    demoted = []

    for seg in local:
        p, info = IWM.fit(family, seg, wmin)
        if not info.get("ok"):
            failures += 1
            demoted.append(seg)
            continue
        lp = IWM.logpmf(family, seg, p, wmin)
        if not np.all(np.isfinite(lp)):
            failures += 1
            demoted.append(seg)
            continue
        nll -= float(lp.sum())
        par_dl += 0.5 * d * np.log(len(seg))
        n_local += 1

    pool = np.concatenate([pooled] + demoted) if demoted else pooled
    n_pooled = int(len(pool))
    if n_pooled:
        p, info = IWM.fit(family, pool, wmin)
        lp = IWM.logpmf(family, pool, p, wmin)
        if not info.get("ok") or not np.all(np.isfinite(lp)):
            return dict(ok=False, reason="pooled group failed", optimizer_failures=failures)
        nll -= float(lp.sum())
        par_dl += 0.5 * d * np.log(n_pooled)

    return dict(ok=True, weight_nll=float(nll), parameter_dl=float(par_dl),
                n_local_groups=n_local, n_pooled_edges=n_pooled,
                optimizer_failures=failures)
