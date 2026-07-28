"""
Nested-fit feasibility probe — which weight likelihoods can be fitted nested, and at what cost?

Context: nested + lognormal is the long-standing blocker. graph-tool 2.98 cannot do weighted
nested at all; 3.0 segfaulted / returned NaN; on 3.5 (with the correct `base_state=` API) it
simply does not return -- >8 h on a 4k-node / 171k-edge subgraph without finishing ONE fit.
An earlier "3.5 nested works" result was an artefact of the WRONG API silently fitting
UNWEIGHTED (all four models returned identical entropies).

So: never launch another blind sweep. Every cell runs in its own subprocess with a HARD
TIMEOUT, so a hang is recorded as TIMEOUT and the probe moves on.

Tests, on the densest `--nodes` subgraph:
  1. each weight model with default minimize settings;
  2. lognormal with tuned settings (the 3.x defaults `B_min_base=1000` and `refine=True` are
     the prime suspects for the hang);
  3. optional size ladder (--ladder) to MEASURE scaling, so full-brain feasibility is
     extrapolated rather than guessed.

Usage:
    python nested_probe.py --edges ~/edges_t5_dir.npz --nodes 1500 --timeout 600
    python nested_probe.py --edges ~/edges_t5_dir.npz --ladder 500,1000,2000 --timeout 900
"""
import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np

MODELS = {"lognormal": (True,  "real-normal"),
          "gaussian":  (False, "real-normal"),
          "poisson":   (False, "discrete-poisson"),
          "geometric": (False, "discrete-geometric")}

VARIANTS = {
    "default":         {},
    "norefine":        dict(refine=False),
    "smallB":          dict(B_min_base=10),
    "norefine+smallB": dict(refine=False, B_min_base=10),
}


def dense_subgraph(edges_path, n_sub, directed=True):
    import graph_tool.all as gt
    d = np.load(edges_path)
    src, dst, w = d["src"].astype(np.int64), d["dst"].astype(np.int64), d["w"].astype(float)
    deg = np.bincount(np.concatenate([src, dst]))
    top = np.argsort(deg)[::-1][:n_sub]
    remap = -np.ones(deg.size, dtype=np.int64)
    remap[top] = np.arange(len(top))
    keep = (remap[src] >= 0) & (remap[dst] >= 0)
    g = gt.Graph(directed=directed)
    g.add_vertex(len(top))
    g.add_edge_list(np.column_stack([remap[src[keep]], remap[dst[keep]]]))
    ew = g.new_ep("double"); ew.a = w[keep]
    lw = g.new_ep("double"); lw.a = np.log(w[keep])
    g.ep["w"], g.ep["logw"] = ew, lw
    return g


def run_one(edges_path, n_sub, model, variant, deg_corr=True):
    import graph_tool.all as gt
    gt.seed_rng(0); np.random.seed(0)
    g = dense_subgraph(edges_path, n_sub)
    use_log, rec_type = MODELS[model]
    prop = g.ep["logw"] if use_log else g.ep["w"]
    kw = dict(VARIANTS[variant])
    t0 = time.time()
    state = gt.minimize_nested_blockmodel_dl(
        g, base_state=gt.WeightedBlockState,
        base_state_args=dict(deg_corr=deg_corr, rec=[prop], rec_types=[rec_type]), **kw)
    ent = float(state.entropy())
    lv = []
    for s in state.get_levels():
        try:
            lv.append(int(s.get_nonempty_B()))
        except Exception:
            lv.append(int(len(np.unique(s.get_blocks().a))))
    return dict(ok=True, entropy=ent, finite=bool(np.isfinite(ent)),
                levels=[b for b in lv if b > 0][:8],
                n_vertices=int(g.num_vertices()), n_edges=int(g.num_edges()),
                secs=round(time.time() - t0, 1))


def probe(a, cells):
    rows = []
    print("\n%-11s %-17s %6s %-10s %14s %16s %7s" %
          ("model", "variant", "nodes", "status", "entropy", "levels", "sec"))
    print("-" * 88)
    for model, variant, n in cells:
        cmd = [sys.executable, __file__, "--edges", a.edges, "--nodes", str(n),
               "--model", model, "--variant", variant, "--child"]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            if line:
                r = json.loads(line[0][7:])
                status = ("OK" if r.get("finite") else "NaN") if r.get("ok") else "ERROR"
            else:
                rc = p.returncode
                status = "SEGFAULT" if rc in (-11, 139) else "CRASH(%s)" % rc
                r = dict(err=(p.stderr or "").strip().splitlines()[-1:])
        except subprocess.TimeoutExpired:
            status, r = "TIMEOUT", dict(secs=a.timeout)
        ent = "%14.1f" % r["entropy"] if r.get("entropy") is not None else " " * 14
        lv = ",".join(map(str, r.get("levels", [])[:5]))
        print("%-11s %-17s %6d %-10s %s %16s %7s" %
              (model, variant, n, status, ent, lv,
               r.get("secs", round(time.time() - t0, 1))), flush=True)
        rows.append(dict(model=model, variant=variant, nodes=n, status=status, **r))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default=os.path.expanduser("~/edges_t5_dir.npz"))
    ap.add_argument("--nodes", type=int, default=1500)
    ap.add_argument("--timeout", type=int, default=600, help="seconds per fit")
    ap.add_argument("--ladder", help="comma sizes, e.g. 500,1000,2000 (lognormal scaling)")
    ap.add_argument("--out", default=os.path.expanduser("~/nested_probe.json"))
    ap.add_argument("--model", default="lognormal")
    ap.add_argument("--variant", default="default")
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()

    if a.child:
        try:
            r = run_one(a.edges, a.nodes, a.model, a.variant)
        except Exception as e:
            r = dict(ok=False, error="%s: %s" % (type(e).__name__, str(e)[:150]))
        print("RESULT " + json.dumps(r), flush=True)
        return

    import graph_tool
    print("graph-tool %s | per-fit timeout %ss" % (graph_tool.__version__, a.timeout))
    cells = []
    if a.ladder:
        for n in [int(x) for x in a.ladder.split(",")]:
            for v in ("default", "norefine+smallB"):
                cells.append(("lognormal", v, n))
    else:
        for m in MODELS:
            cells.append((m, "default", a.nodes))
        for v in ("norefine", "smallB", "norefine+smallB"):
            cells.append(("lognormal", v, a.nodes))
    rows = probe(a, cells)
    json.dump(rows, open(a.out, "w"), indent=2)
    ok = [r for r in rows if r["status"] == "OK"]
    print("\n%d/%d fits completed. saved -> %s" % (len(ok), len(rows), a.out))
    if ok:
        print("usable: " + ", ".join("%s/%s(%ss)" % (r["model"], r["variant"], r["secs"])
                                     for r in ok))


if __name__ == "__main__":
    main()
