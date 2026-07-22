"""
Degree-corrected SBM inference using graph-tool's OWN minimizer (graph-tool 2.98). No
homemade search.

Weight likelihoods are edge covariates passed as recs on BlockState (the 2.x API):
  lognormal = real-normal on log-weights ; gaussian = real-normal on raw weights ;
  poisson / geometric = discrete on raw weights.
"""
import graph_tool.all as gt
import numpy as np
from graph import finest_blocks

WEIGHT_MODELS = {
    "lognormal":  ("logw", "real-normal"),
    "gaussian":   ("w",    "real-normal"),
    "poisson":    ("w",    "discrete-poisson"),
    "geometric":  ("w",    "discrete-geometric"),
    "unweighted": (None,   None),
}


def fit(g, model, nested=False, deg_corr=True, seed=0):
    """Fit a (nested) weighted DC-SBM with graph-tool's minimizer. Returns (state, info)."""
    gt.seed_rng(seed)
    np.random.seed(seed)
    prop_name, rec_type = WEIGHT_MODELS[model]
    sargs = dict(deg_corr=deg_corr)
    if prop_name is not None:
        sargs.update(recs=[g.ep[prop_name]], rec_types=[rec_type])
    if nested:
        state = gt.minimize_nested_blockmodel_dl(g, state_args=sargs)
    else:
        state = gt.minimize_blockmodel_dl(g, state=gt.BlockState, state_args=sargs)
    info = dict(model=model, nested=nested, deg_corr=deg_corr, seed=seed,
                entropy=float(state.entropy()),
                n_blocks=int(len(np.unique(finest_blocks(state)))))
    return state, info


def blocks_by_neuron(state, node_ids):
    """Map the finest-level block assignment back to neuron ids: {neuron_id: block}."""
    b = finest_blocks(state)
    return {int(n): int(bb) for n, bb in zip(node_ids, b)}
