"""
Degree-corrected SBM inference on graph-tool 3.0, run as a MONITORED, INTERRUPTIBLE loop.

Instead of the opaque high-level minimizer (the run that hung for weeks), we drive the same
production-quality move -- `multilevel_mcmc_sweep` (parallel across cores) -- one sweep at a
time, calling the Monitor after each: heartbeat, wall-clock timeout, atomic checkpoints,
multi-signal convergence. An external watchdog (monitor.run_supervised) can still SIGKILL a
sweep that hangs; a SIGTERM handler here turns the watchdog's graceful phase into a clean
checkpoint + TIMED_OUT before that.

Weight likelihoods use WeightedBlockState(rec=[prop], rec_types=[name]):
  "real-normal"  on the LOG-weight property  -> lognormal
  "real-normal"  on the RAW-weight property  -> Gaussian
  "discrete-poisson" / "discrete-geometric"  on the RAW-weight property
"""
import random
import signal
import time
import graph_tool.all as gt
import numpy as np
from monitor import Monitor, Timeout, load_checkpoint
from graph import finest_blocks, graph_checksum

WEIGHT_MODELS = {
    "lognormal":  ("logw", "real-normal"),
    "gaussian":   ("w",    "real-normal"),
    "poisson":    ("w",    "discrete-poisson"),
    "geometric":  ("w",    "discrete-geometric"),
    "unweighted": (None,   None),
}


def make_state(g, model, nested=True, deg_corr=True, init_blocks=None):
    """Build a (nested) weighted DC-SBM state; optionally warm-start from saved blocks.
    init_blocks is the per-level list from a checkpoint (['blocks'])."""
    prop_name, rec_type = WEIGHT_MODELS[model]
    sargs = dict(deg_corr=deg_corr)
    if prop_name is not None:
        sargs.update(rec=[g.ep[prop_name]], rec_types=[rec_type])
    if nested:
        kw = dict(base_state=gt.WeightedBlockState, base_state_args=sargs)
        if init_blocks is not None:
            kw["bs"] = init_blocks
        return gt.NestedBlockState(g, **kw)
    if init_blocks is not None:
        return gt.WeightedBlockState(g, b=init_blocks[0], **sargs)
    return gt.WeightedBlockState(g, **sargs)


def fit(g, model, run_name, nested=True, deg_corr=True, seed=0,
        max_seconds=6 * 3600, patience=25, sweep_niter=1, beta=float("inf"),
        checkpoint_s=600, heartbeat_s=15, resume=False):
    """
    Fit a DC-SBM and return (state, info). Never hangs (in-loop timeout + external
    watchdog). A TIMED_OUT status must not be treated as converged.
    """
    gt.seed_rng(seed)
    np.random.seed(seed)
    cfg = dict(model=model, nested=nested, deg_corr=deg_corr, directed=bool(g.is_directed()))

    init_blocks = None
    if resume:
        ckpt = load_checkpoint(run_name)
        if ckpt is not None:
            if ckpt["graph_checksum"] != graph_checksum(g):
                raise ValueError(f"resume {run_name}: graph checksum mismatch")
            init_blocks = ckpt["blocks"]
            np.random.set_state(ckpt["rng"]["numpy"])
            random.setstate(ckpt["rng"]["python"])
            print(f"[sbm] resume {run_name}: from iter {ckpt['iter']} S={ckpt['entropy']:.1f} "
                  f"(continues from partition; not bit-identical)")

    state = make_state(g, model, nested=nested, deg_corr=deg_corr, init_blocks=init_blocks)
    mon = Monitor(run_name, seed, max_seconds, model_config=cfg,
                  checkpoint_s=checkpoint_s, heartbeat_s=heartbeat_s)

    # SIGTERM (from the external watchdog's graceful phase) -> stop cleanly after this sweep
    stop = {"flag": False}
    old_handler = signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))

    S = float(state.entropy())          # exact once; then track incrementally via dS (cheap)
    best_S, no_improve, status = S, 0, "CONVERGED"
    try:
        while True:
            dS, n_att, n_moves = state.multilevel_mcmc_sweep(
                niter=sweep_niter, beta=beta, parallel=True)
            S += dS
            mon.record(state, S, n_moves / max(n_att, 1))
            if S < best_S - 1e-6:
                best_S, no_improve = S, 0
            else:
                no_improve += 1
            if stop["flag"]:
                status = "TIMED_OUT"; break
            if no_improve >= patience or mon.converged():
                break
    except Timeout:
        status = "TIMED_OUT"
    finally:
        signal.signal(signal.SIGTERM, old_handler)

    S_final = float(state.entropy())
    mon.finish(status, state=state, S=S_final)
    info = dict(model=model, nested=nested, deg_corr=deg_corr, seed=seed, status=status,
                entropy=S_final, n_blocks=int(len(np.unique(finest_blocks(state)))),
                iters=mon._iter, elapsed_s=round(time.time() - mon.t0, 1))
    print(f"[sbm] {run_name}: {status}  S={S_final:.1f}  blocks={info['n_blocks']}  "
          f"iters={info['iters']}  {info['elapsed_s']}s")
    return state, info


def blocks_by_neuron(state, node_ids):
    """Map the finest-level block assignment back to neuron ids: {neuron_id: block}."""
    b = finest_blocks(state)
    return {int(n): int(bb) for n, bb in zip(node_ids, b)}
