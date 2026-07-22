"""
Leak-free held-out predictive scoring of weight likelihoods.

SCORER (fixed, correct now): every family is scored as a TRUNCATED DISCRETE pmf on the same
integer support -- log P(W=k | W>=threshold) -- so nats/edge are comparable across
lognormal / Gaussian / Poisson / geometric. The lognormal and Gaussian continuous densities
are discretized via  P(W=k)=F(k+0.5)-F(k-0.5), conditioned on W>=threshold. Without this the
cross-family numbers are on different measures and meaningless (reviewer P1 #4).

FOLD (PROVISIONAL -- pending the Gate A-1 feasibility probe): the current fold removes test
edges before fitting (so the partition is leak-free) and scores only their conditional weight.
That is neither the plan's "pure weight prediction" (which keeps adjacency, masks only the
weight) nor the "joint edge prediction" fallback (which also scores edge existence with
non-edge sampling + importance weighting). Which of those we implement is decided by the A-1
probe (can graph-tool mask a weight without leaking it?). Until then, treat the fold's output
as a leak-freedom / plumbing check, NOT as an E2b model-selection result.
"""
import numpy as np
from scipy.stats import norm, poisson, geom
import graph as G
import sbm


def split_edges(m, test_frac, seed):
    """Split the ORIGINAL edge arrays (pre/post/weight index order) into train/test."""
    rng = np.random.default_rng(seed)
    test_idx = np.sort(rng.choice(m, size=int(round(test_frac * m)), replace=False))
    train_mask = np.ones(m, dtype=bool)
    train_mask[test_idx] = False
    return train_mask, test_idx


def train_graph(pre, post, weight, train_mask, directed):
    """Graph from TRAINING edges only (held-out edges physically absent -> leak-free partition)."""
    return G.build_graph(pre[train_mask], post[train_mask], weight[train_mask], directed=directed)


# ---- common-support scorer ---------------------------------------------------
def _fit_params(model, w_train):
    x = np.asarray(w_train, dtype=float)
    if model == "lognormal":
        lx = np.log(x); return {"mu": lx.mean(), "sigma": max(lx.std(ddof=0), 1e-6)}
    if model == "gaussian":
        return {"mu": x.mean(), "sigma": max(x.std(ddof=0), 1e-6)}
    if model == "poisson":
        return {"lam": max(x.mean(), 1e-9)}
    if model == "geometric":                       # support k>=1
        return {"p": min(max(1.0 / max(x.mean(), 1.0), 1e-6), 1 - 1e-6)}
    raise ValueError(model)


def _log_trunc_pmf(model, k, p, threshold):
    """log P(W=k) - log P(W>=threshold), integer support, common across families."""
    if model in ("lognormal", "gaussian"):
        if model == "lognormal":
            z = lambda x: norm.cdf((np.log(x) - p["mu"]) / p["sigma"])
        else:
            z = lambda x: norm.cdf((x - p["mu"]) / p["sigma"])
        pk = z(k + 0.5) - z(k - 0.5)
        tail = 1.0 - z(threshold - 0.5)
    elif model == "poisson":
        pk = poisson.pmf(k, p["lam"]); tail = poisson.sf(threshold - 1, p["lam"])
    elif model == "geometric":
        pk = geom.pmf(k, p["p"]); tail = geom.sf(threshold - 1, p["p"])
    else:
        raise ValueError(model)
    return float(np.log(max(pk, 1e-300)) - np.log(max(tail, 1e-300)))


def predictive_logscore(state, node_ids, pre, post, weight, test_idx, train_mask,
                        model, threshold):
    """Mean truncated-discrete predictive log-score (nats/edge) on held-out edges.
    Block assignment + per-block-pair params come from TRAIN only; held-out weights are
    scored, never fitted."""
    from collections import defaultdict
    blk = sbm.blocks_by_neuron(state, node_ids)
    w = weight.astype(float)

    pair_w = defaultdict(list)
    for i in np.nonzero(train_mask)[0]:
        a, b = blk.get(int(pre[i])), blk.get(int(post[i]))
        if a is not None and b is not None:
            pair_w[(a, b)].append(w[i])
    global_p = _fit_params(model, w[train_mask])
    pair_p = {k: _fit_params(model, v) for k, v in pair_w.items() if len(v) >= 5}

    scores = []
    for i in test_idx:
        a, b = blk.get(int(pre[i])), blk.get(int(post[i]))
        scores.append(_log_trunc_pmf(model, int(weight[i]), pair_p.get((a, b), global_p), threshold))
    scores = np.array(scores)
    return float(scores.mean()), scores


def run_fold(pre, post, weight, directed, model, threshold, test_frac=0.1, seed=0,
             fit_seconds=1800, nested=True):
    """One provisional leak-free fold end-to-end (plumbing/leak check, not an E2b result)."""
    train_mask, test_idx = split_edges(len(weight), test_frac, seed)
    g_tr, node_ids, _ = train_graph(pre, post, weight, train_mask, directed)
    assert g_tr.num_edges() == int(train_mask.sum()), "train graph leaked held-out edges"
    state, info = sbm.fit(g_tr, model, run_name=f"xval_{model}_seed{seed}",
                          nested=nested, seed=seed, max_seconds=fit_seconds)
    if info["status"] != "CONVERGED":
        # never score a timed-out / non-converged fit
        return {"model": model, "seed": seed, "fit_status": info["status"], "score": None}
    score, _ = predictive_logscore(state, node_ids, pre, post, weight,
                                   test_idx, train_mask, model, threshold)
    return {"model": model, "seed": seed, "test_frac": test_frac, "n_test": len(test_idx),
            "logscore_per_edge": score, "fit_status": info["status"], "n_blocks": info["n_blocks"]}
