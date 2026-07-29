"""
graph-tool 3.5 regression test — re-run the exact failures that forced us off 3.0/3.1.

Original failures (RESULTS/PLAN):
  F1  NESTED weighted fit SEGFAULTS (returncode -11) on the real connectome.
  F2  FLAT weighted fit returns entropy = NaN (real-normal microcanonical default),
      and block count explodes.

This script runs, per weight model x {DC, non-DC} x {directed, undirected}:
  - flat fit    -> entropy finite? n_blocks sane?
  - nested fit  -> completes? entropy finite? how many levels?
on a subgraph (default 2000 densest nodes) so a crash is cheap and fast.

Each cell runs in a SUBPROCESS so a segfault is caught and reported, not fatal.
Usage:  python test_gt35.py --nodes 2000 [--models lognormal,poisson] [--cell <spec>]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

WEIGHT_MODELS = {
    "lognormal":   ("logw", "real-normal"),
    "gaussian":    ("w",    "real-normal"),
    "poisson":     ("w",    "discrete-poisson"),
    "geometric":   ("w",    "discrete-geometric"),
    "exponential": ("w",    "real-exponential"),
}


def build_subgraph(n_nodes, directed):
    """Real connectome subgraph: densest n_nodes, with w and logw edge properties."""
    import numpy as np
    import graph_tool.all as gt
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import data
    pre, post, w = data.load_edges(threshold=5, directed=directed)
    vals, counts = np.unique(np.concatenate([pre, post]), return_counts=True)
    top = vals[np.argsort(counts)[::-1][:n_nodes]]
    m = np.isin(pre, top) & np.isin(post, top)
    pre, post, w = pre[m], post[m], w[m].astype(float)
    ids = np.unique(np.concatenate([pre, post]))
    idx = {int(v): i for i, v in enumerate(ids)}
    g = gt.Graph(directed=directed)
    g.add_vertex(len(ids))
    g.add_edge_list(np.column_stack([[idx[int(a)] for a in pre],
                                     [idx[int(b)] for b in post]]))
    ew = g.new_ep("double"); ew.a = w
    lw = g.new_ep("double"); lw.a = np.log(w)
    g.ep["w"], g.ep["logw"] = ew, lw
    return g


def run_one(model, nested, deg_corr, directed, n_nodes, seed=0):
    """The actual fit (runs inside the subprocess)."""
    import numpy as np
    import graph_tool.all as gt
    gt.seed_rng(seed); np.random.seed(seed)
    g = build_subgraph(n_nodes, directed)
    prop, rtype = WEIGHT_MODELS[model]
    sargs = dict(deg_corr=deg_corr, recs=[g.ep[prop]], rec_types=[rtype])
    t0 = time.time()
    if nested:
        state = gt.minimize_nested_blockmodel_dl(g, state_args=sargs)
        levels = [s.get_nonempty_B() for s in state.get_levels()]
        ent = float(state.entropy())
        nb = int(levels[0])
    else:
        state = gt.minimize_blockmodel_dl(g, state=gt.BlockState, state_args=sargs)
        levels = None
        ent = float(state.entropy())
        nb = int(state.get_nonempty_B())
    return dict(ok=True, entropy=ent, finite=bool(np.isfinite(ent)), n_blocks=nb,
                levels=[int(x) for x in levels] if levels else None,
                n_vertices=int(g.num_vertices()), n_edges=int(g.num_edges()),
                secs=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=2000)
    ap.add_argument("--models", default="lognormal,gaussian,poisson,geometric,exponential")
    ap.add_argument("--cell", help="internal: model,nested,dc,directed")
    a = ap.parse_args()

    if a.cell:                                   # child process: run one fit, print JSON
        model, nested, dc, directed = a.cell.split(",")
        try:
            r = run_one(model, nested == "1", dc == "1", directed == "1", a.nodes)
        except Exception as e:
            r = dict(ok=False, error=f"{type(e).__name__}: {e}")
        print("RESULT " + json.dumps(r))
        return

    models = a.models.split(",")
    print(f"graph-tool regression test — subgraph {a.nodes} nodes\n")
    import graph_tool
    print(f"graph-tool version: {graph_tool.__version__}\n")
    print(f"{'model':12} {'fit':7} {'dc':4} {'dir':4} {'status':10} {'entropy':>14} "
          f"{'blocks':>7} {'levels':>8} {'sec':>6}")
    results = []
    for model in models:
        for nested in (False, True):
            for dc in (True, False):
                for directed in (True, False):
                    spec = f"{model},{int(nested)},{int(dc)},{int(directed)}"
                    p = subprocess.run(
                        [sys.executable, __file__, "--nodes", str(a.nodes), "--cell", spec],
                        capture_output=True, text=True)
                    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
                    if p.returncode != 0 and not line:
                        status = f"CRASH({p.returncode})"          # -11 = segfault
                        r = dict(ok=False, crash=p.returncode)
                    elif line:
                        r = json.loads(line[0][7:])
                        if not r.get("ok"):
                            status = "ERROR"
                        elif not r.get("finite"):
                            status = "NaN-ENT"
                        else:
                            status = "OK"
                    else:
                        status = "NO-OUTPUT"; r = dict(ok=False)
                    ent = f"{r['entropy']:14.1f}" if r.get("entropy") is not None else " " * 14
                    lv = str(r.get("levels"))[:8] if r.get("levels") else ""
                    print(f"{model:12} {'nested' if nested else 'flat':7} "
                          f"{'dc' if dc else 'ndc':4} {'dir' if directed else 'und':4} "
                          f"{status:10} {ent} {str(r.get('n_blocks','')):>7} {lv:>8} "
                          f"{str(r.get('secs','')):>6}", flush=True)
                    r.update(model=model, nested=nested, dc=dc, directed=directed, status=status)
                    results.append(r)
    out = Path("/var/tmp/csbm_work/gt35_regression.json")
    out.write_text(json.dumps(results, indent=2))
    n_ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\n{n_ok}/{len(results)} cells OK -> {out}")
    nested_ok = sum(1 for r in results if r["nested"] and r["status"] == "OK")
    nested_n = sum(1 for r in results if r["nested"])
    print(f"NESTED specifically: {nested_ok}/{nested_n} OK "
          f"(this is what failed on 3.0: segfault; and NaN on flat)")


if __name__ == "__main__":
    main()
