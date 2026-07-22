"""
Run ONE SBM fit as a supervised child process.

Reviewer P1 #1: the in-loop timeout only fires *between* sweeps, so a sweep that hangs is
uninterruptible in-process. The fix is to run each fit as a child of monitor.run_supervised,
which can SIGTERM (this process checkpoints via the SIGTERM handler in sbm.fit) and then
SIGKILL even during a hung C++ sweep. This script is that child.

Usage (real):  python worker.py --run-name r0 --model lognormal --nested --seed 0
Usage (smoke): python worker.py --run-name smoke --model lognormal --nested --test
Run with cwd = src/ (bare intra-package imports).
"""
import argparse
import json
import sys
import numpy as np
import graph_tool.all as gt
import config
import data
import graph
import sbm


def build_test_graph(seed=0):
    """Tiny built-in graph with synthetic integer weights, for smoke-testing only."""
    g = gt.collection.data["polbooks"].copy()
    rng = np.random.default_rng(seed)
    w = rng.integers(1, 30, g.num_edges()).astype(float)
    w_ep = g.new_ep("double"); w_ep.a = w
    logw_ep = g.new_ep("double"); logw_ep.a = np.log(w)
    g.ep["w"] = w_ep
    g.ep["logw"] = logw_ep
    return g, np.arange(g.num_vertices())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--model", default="lognormal", choices=list(sbm.WEIGHT_MODELS))
    ap.add_argument("--nested", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=6 * 3600)
    ap.add_argument("--threshold", type=int, default=config.DEFAULT_THRESHOLD)
    ap.add_argument("--undirected", action="store_true")
    ap.add_argument("--checkpoint-s", type=float, default=600)
    ap.add_argument("--heartbeat-s", type=float, default=15)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--test", action="store_true", help="use a tiny built-in graph")
    a = ap.parse_args()

    directed = not a.undirected
    if a.test:
        g, _ = build_test_graph(a.seed)
    else:
        pre, post, w = data.load_edges(threshold=a.threshold, directed=directed)
        g, _, _ = graph.build_graph(pre, post, w, directed=directed)

    _, info = sbm.fit(g, a.model, run_name=a.run_name, nested=a.nested, seed=a.seed,
                      max_seconds=a.max_seconds, checkpoint_s=a.checkpoint_s,
                      heartbeat_s=a.heartbeat_s, resume=a.resume)
    print(json.dumps(info))
    sys.exit(0 if info["status"] == "CONVERGED" else 2)


if __name__ == "__main__":
    main()
