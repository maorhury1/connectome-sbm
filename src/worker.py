"""
Run ONE SBM fit and SAVE ALL RAW OUTPUTS (no biology evaluation yet):
  - partition           (neuron id -> block)
  - MDL / entropy        (the description length the algorithm minimises)
  - block-pair matrices  (edge count, weight sum, weight sum-of-squares) => block weight
                         means/variances are recoverable
  - block sizes, n_blocks, n_nodes/edges, elapsed, full config
Run as a child of the scheduler so it can be killed if it hangs. Run with cwd = src/.
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import graph_tool.all as gt
import config
import data
import graph
import sbm
from graph import finest_blocks

RESULTS_DIR = config.WORK_DIR / "results"


def build_test_graph(seed=0):
    g = gt.collection.data["polbooks"].copy()
    rng = np.random.default_rng(seed)
    w = rng.integers(1, 30, g.num_edges()).astype(float)
    w_ep = g.new_ep("double"); w_ep.a = w
    logw_ep = g.new_ep("double"); logw_ep.a = np.log(w)
    g.ep["w"] = w_ep; g.ep["logw"] = logw_ep
    return g, np.arange(g.num_vertices())


def block_matrices(g, fb):
    """Block-pair edge count, weight sum, weight sum-of-squares (compact, relabelled blocks)."""
    uniq, inv = np.unique(fb, return_inverse=True)
    K = len(uniq)
    E = g.get_edges()
    w = g.ep["w"].a
    bs, bt = inv[E[:, 0]], inv[E[:, 1]]
    flat = bs * K + bt
    ecount = np.bincount(flat, minlength=K * K).reshape(K, K)
    wsum = np.bincount(flat, weights=w, minlength=K * K).reshape(K, K)
    wsq = np.bincount(flat, weights=w * w, minlength=K * K).reshape(K, K)
    return uniq, ecount, wsum, wsq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--model", default="lognormal", choices=list(sbm.WEIGHT_MODELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=int, default=1)
    ap.add_argument("--undirected", action="store_true")
    ap.add_argument("--no-dc", action="store_true")
    ap.add_argument("--nested", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()

    directed = not a.undirected
    cfg = dict(model=a.model, seed=a.seed, threshold=a.threshold,
               directed=directed, deg_corr=not a.no_dc, nested=a.nested)

    if a.test:
        g, node_ids = build_test_graph(a.seed)
    else:
        pre, post, w = data.load_edges(threshold=a.threshold, directed=directed)
        g, node_ids, _ = graph.build_graph(pre, post, w, directed=directed)

    t0 = time.time()
    state, info = sbm.fit(g, a.model, nested=a.nested, deg_corr=not a.no_dc, seed=a.seed)
    elapsed = time.time() - t0

    fb = finest_blocks(state)
    uniq, ecount, wsum, wsq = block_matrices(g, fb)
    sizes = np.bincount(np.unique(fb, return_inverse=True)[1])

    outdir = RESULTS_DIR / a.run_name
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / "partition.npz", node_ids=node_ids, blocks=fb)
    np.savez_compressed(outdir / "blockmat.npz",
                        block_ids=uniq, ecount=ecount, wsum=wsum, wsq=wsq)
    summary = dict(config=cfg, status="OK",
                   mdl_entropy=float(state.entropy()),
                   n_blocks=int(info["n_blocks"]),
                   n_nodes=int(g.num_vertices()), n_edges=int(g.num_edges()),
                   block_sizes=dict(n=int(len(sizes)), min=int(sizes.min()),
                                    max=int(sizes.max()), median=float(np.median(sizes)),
                                    n_singletons=int((sizes == 1).sum())),
                   elapsed_s=round(elapsed, 1))
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    sys.exit(0)


if __name__ == "__main__":
    main()
