"""
Leak-free held-out predictive scoring of weight likelihoods (Gate A-1 core).

The plan's non-negotiable: the PARTITION and ALL fitted parameters must come from TRAINING
data only; held-out values may enter *solely* at final scoring. We guarantee this
structurally: held-out edges are physically REMOVED from the graph before inference, so the
block assignment cannot see them (neither adjacency nor weight). Held-out weights are then
scored using per-block-pair predictive parameters estimated from TRAINING edges only.

This is the "joint edge prediction" fold (adjacency + weight both hidden) — provably
leak-free. Whether graph-tool can instead keep an edge visible while masking *only* its
weight (pure "weight prediction") is the open A-1 question the feasibility test probes; if it
cannot be done without leakage, this module is the predeclared fallback (outcome 2 in PLAN).

Scores are natural-scale predictive log-densities per held-out edge (nats/edge). The
common-integer-support discretization for cross-FAMILY comparison is layered on in E2b; here
we only need to show a leak-free fold yields a valid per-model score.
"""
import numpy as np
from scipy import stats
import graph as G
import sbm


def split_edges(m, test_frac, seed):
    """Split the ORIGINAL edge arrays (pre/post/weight index order) into train/test.
    Returns (train_mask, test_idx)."""
    rng = np.random.default_rng(seed)
    test_idx = np.sort(rng.choice(m, size=int(round(test_frac * m)), replace=False))
    train_mask = np.ones(m, dtype=bool)
    train_mask[test_idx] = False
    return train_mask, test_idx


def train_graph(pre, post, weight, train_mask, directed):
    """Build a graph from TRAINING edges only (held-out edges absent)."""
    return G.build_graph(pre[train_mask], post[train_mask], weight[train_mask], directed=directed)


def _pair_params(model, logw_or_w):
    """Fit a weight model's parameters to a 1-D array of TRAIN weights in one block-pair."""
    x = np.asarray(logw_or_w, dtype=float)
    if model in ("lognormal", "gaussian"):
        return {"mu": x.mean(), "var": max(x.var(ddof=0), 1e-9)}
    if model == "poisson":
        return {"lam": max(x.mean(), 1e-9)}
    if model == "geometric":                       # support k>=1; p = 1/mean
        return {"p": min(max(1.0 / max(x.mean(), 1.0), 1e-6), 1 - 1e-6)}
    raise ValueError(model)


def _logpdf(model, value, params):
    """Predictive log-density/pmf of a single held-out weight under fitted params."""
    if model in ("lognormal", "gaussian"):
        return stats.norm.logpdf(value, loc=params["mu"], scale=np.sqrt(params["var"]))
    if model == "poisson":
        return stats.poisson.logpmf(value, mu=params["lam"])
    if model == "geometric":
        return stats.geom.logpmf(value, p=params["p"])   # k in {1,2,...}
    raise ValueError(model)


def predictive_logscore(state, node_ids, pre, post, weight, test_idx, train_mask, model):
    """
    Mean held-out predictive log-score (nats/edge). Block assignment comes from the
    TRAIN-fitted `state`; block-pair params are estimated from TRAIN edges only; held-out
    weights are scored, never fitted.
    'lognormal' scores log-weights; the other models score raw weights.
    """
    blk = sbm.blocks_by_neuron(state, node_ids)
    use_log = (model == "lognormal")
    val = np.log(weight.astype(float)) if use_log else weight.astype(float)

    # accumulate TRAIN values per ordered block-pair
    from collections import defaultdict
    pair_vals = defaultdict(list)
    tr = np.nonzero(train_mask)[0]
    for i in tr:
        a, b = blk.get(int(pre[i])), blk.get(int(post[i]))
        if a is None or b is None:
            continue
        pair_vals[(a, b)].append(val[i])
    global_params = _pair_params(model, val[train_mask])           # fallback for unseen pairs
    pair_params = {k: _pair_params(model, v) for k, v in pair_vals.items() if len(v) >= 5}

    scores = []
    for i in test_idx:
        a, b = blk.get(int(pre[i])), blk.get(int(post[i]))
        p = pair_params.get((a, b), global_params)
        scores.append(_logpdf(model, val[i], p))
    scores = np.array(scores, dtype=float)
    return float(scores.mean()), scores


def run_fold(pre, post, weight, directed, model, test_frac=0.1, seed=0,
             fit_seconds=1800, nested=True):
    """One leak-free CV fold end-to-end. Returns dict with the held-out score and a
    leak-freedom assertion (train graph excludes every test edge)."""
    train_mask, test_idx = split_edges(len(weight), test_frac, seed)
    g_tr, node_ids, _ = train_graph(pre, post, weight, train_mask, directed)
    # leak-freedom check: the fit graph must contain exactly the training edges
    assert g_tr.num_edges() == int(train_mask.sum()), "train graph leaked held-out edges"
    state, info = sbm.fit(g_tr, model, run_name=f"xval_{model}_seed{seed}",
                          nested=nested, seed=seed, max_seconds=fit_seconds)
    score, per_edge = predictive_logscore(state, node_ids, pre, post, weight,
                                          test_idx, train_mask, model)
    return {"model": model, "seed": seed, "test_frac": test_frac,
            "n_test": len(test_idx), "logscore_per_edge": score,
            "fit_status": info["status"], "n_blocks": info["n_blocks"]}
