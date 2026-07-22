"""
Degree-corrected SBM inference on graph-tool 3.0, run as a MONITORED, INTERRUPTIBLE loop.

Why a manual sweep loop instead of minimize_blockmodel_dl:
The high-level minimizer is a single opaque call (the run that hung for weeks). Instead we
drive the *same* production-quality move — `multilevel_mcmc_sweep` (parallel across cores) —
one sweep at a time, calling the Monitor after each. This gives heartbeats, wall-clock
timeout, atomic checkpoints, and multi-signal convergence on the real path.

Weight likelihoods use WeightedBlockState(rec=[prop], rec_types=[name]):
  "real-normal"  on the LOG-weight property  -> lognormal
  "real-normal"  on the RAW-weight property  -> Gaussian
  "discrete-poisson" / "discrete-geometric" on the RAW-weight property
"""
import graph_tool.all as gt
import numpy as np
from monitor import Monitor, Timeout

# maps our short model names to (edge-property-name, graph-tool rec_type)
WEIGHT_MODELS = {
    "lognormal":  ("logw", "real-normal"),
    "gaussian":   ("w",    "real-normal"),
    "poisson":    ("w",    "discrete-poisson"),
    "geometric":  ("w",    "discrete-geometric"),
    "unweighted": (None,   None),
}


def make_state(g, model, nested=True, deg_corr=True):
    """Build a (nested) weighted DC-SBM state for a given weight model."""
    prop_name, rec_type = WEIGHT_MODELS[model]
    sargs = dict(deg_corr=deg_corr)
    if prop_name is not None:
        sargs.update(rec=[g.ep[prop_name]], rec_types=[rec_type])
    if nested:
        return gt.NestedBlockState(g, base_state=gt.WeightedBlockState, base_state_args=sargs)
    return gt.WeightedBlockState(g, **sargs)


def fit(g, model, run_name, nested=True, deg_corr=True, seed=0,
        max_seconds=6 * 3600, patience=25, sweep_niter=1, beta=float("inf"),
        checkpoint_s=600, heartbeat_s=15):
    """
    Fit a DC-SBM and return (state, info). Never hangs: bounded by max_seconds; on breach
    the partial state is checkpointed and status='TIMED_OUT' (caller must not treat a
    timed-out fit as converged).
    """
    gt.seed_rng(seed)
    np.random.seed(seed)
    state = make_state(g, model, nested=nested, deg_corr=deg_corr)
    mon = Monitor(run_name, seed, max_seconds, checkpoint_s=checkpoint_s, heartbeat_s=heartbeat_s)

    S = float(state.entropy())          # exact once; then track incrementally via dS (cheap)
    best_S = S
    no_improve = 0
    status = "CONVERGED"
    try:
        while True:
            dS, n_att, n_moves = state.multilevel_mcmc_sweep(
                niter=sweep_niter, beta=beta, parallel=True)
            S += dS
            accept = n_moves / max(n_att, 1)
            mon.record(state, S, accept)
            if S < best_S - 1e-6:
                best_S = S
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience or mon.converged():
                break
    except Timeout:
        status = "TIMED_OUT"

    S_final = float(state.entropy())    # exact recompute at the end
    mon.finish(status, state=state, S=S_final)
    info = {"model": model, "nested": nested, "deg_corr": deg_corr, "seed": seed,
            "status": status, "entropy": S_final, "n_blocks": _nblocks(state),
            "iters": mon._iter, "elapsed_s": mon._S_hist and (mon.t0 and None) or None}
    print(f"[sbm] {run_name}: {status}  S={S_final:.1f}  blocks={info['n_blocks']}  iters={mon._iter}")
    return state, info


def _nblocks(state):
    if hasattr(state, "get_bs"):
        return int(len(np.unique(np.asarray(state.get_bs()[0].a))))
    return int(len(np.unique(np.asarray(state.get_blocks().a))))


def blocks_by_neuron(state, node_ids):
    """Map the finest-level block assignment back to neuron ids: {neuron_id: block}."""
    b = state.get_bs()[0].a if hasattr(state, "get_bs") else state.get_blocks().a
    b = np.asarray(b)
    return {int(n): int(bb) for n, bb in zip(node_ids, b)}
