"""Run ONE SBM fit as a child process, so monitor.run_supervised can kill it if it hangs.
Prints the result as one JSON line. Run with cwd = src/."""
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
    ap.add_argument("--model", default="lognormal", choices=list(sbm.WEIGHT_MODELS))
    ap.add_argument("--nested", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=int, default=config.DEFAULT_THRESHOLD)
    ap.add_argument("--undirected", action="store_true")
    ap.add_argument("--test", action="store_true", help="use a tiny built-in graph")
    a = ap.parse_args()

    directed = not a.undirected
    if a.test:
        g, _ = build_test_graph(a.seed)
    else:
        pre, post, w = data.load_edges(threshold=a.threshold, directed=directed)
        g, _, _ = graph.build_graph(pre, post, w, directed=directed)

    _, info = sbm.fit(g, a.model, nested=a.nested, seed=a.seed)
    print(json.dumps(info))
    sys.exit(0)


if __name__ == "__main__":
    main()
