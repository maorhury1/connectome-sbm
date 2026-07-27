"""
NESTED SBM sweep — the hierarchy experiment (RQ-D / E4), portable.

Grid: 4 weight models x {DC, non-DC} x {directed, undirected} = 16 nested fits on the canonical
>=5 connectome, from the .npz written by export_edges.py.

REQUIRES graph-tool >= 3.1. On graph-tool 2.98 the weighted NESTED fit does not work, and 3.1+
cannot run on the lab server (needs glibc 2.38, server has 2.35) — so this script is meant to be
run on a machine that can (macOS, or an Ubuntu 24.04+ host). Verified on graph-tool 3.5 / macOS
arm64: 16/16 nested cells succeed on synthetic data, including nested + lognormal + DC.

Robustness (learned the hard way):
  - every cell runs in its OWN subprocess -> a segfault is recorded, not fatal;
  - every finished cell writes its own JSON + partition .npz -> re-running SKIPS completed cells,
    so an interrupted overnight run resumes for free;
  - cells run in PRIORITY order (lognormal+DC first), so if only some finish, they are the ones
    that matter.

Usage:
    python nested_sweep.py --edges edges_t5_dir.npz --out nested_results --jobs 3
    python report_nested.py --out nested_results        # table when done (or partway)
"""
import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np

WEIGHT_MODELS = {
    "lognormal": (True,  "real-normal"),
    "gaussian":  (False, "real-normal"),
    "poisson":   (False, "discrete-poisson"),
    "geometric": (False, "discrete-geometric"),
}
PRIORITY = ["lognormal", "geometric", "poisson", "gaussian"]   # key model first


def n_blocks_of(state):
    """Block count across graph-tool versions (3.5 dropped get_nonempty_B on WeightedBlockState)."""
    for m in ("get_nonempty_B", "get_B"):
        if hasattr(state, m):
            try:
                return int(getattr(state, m)())
            except Exception:
                pass
    return int(len(np.unique(state.get_blocks().a)))


def build_graph(edges_path, directed):
    """Graph from the exported edge list; undirected collapses reciprocal pairs (sums weights)."""
    import graph_tool.all as gt
    d = np.load(edges_path)
    src, dst = d["src"].astype(np.int64), d["dst"].astype(np.int64)
    w = d["w"].astype(float)
    n = int(max(src.max(), dst.max())) + 1
    if not directed:
        a = np.minimum(src, dst)
        b = np.maximum(src, dst)
        key = a * n + b
        uk, inv = np.unique(key, return_inverse=True)
        w = np.bincount(inv, weights=w)
        src, dst = uk // n, uk % n
    g = gt.Graph(directed=directed)
    g.add_vertex(n)
    g.add_edge_list(np.column_stack([src, dst]))
    ew = g.new_ep("double"); ew.a = w
    lw = g.new_ep("double"); lw.a = np.log(w)
    g.ep["w"], g.ep["logw"] = ew, lw
    return g, d["node_ids"]


def run_cell(model, deg_corr, directed, seed, edges_path, out_dir, name):
    """One nested fit; saves the finest-level partition next to the summary."""
    import graph_tool.all as gt
    gt.seed_rng(seed); np.random.seed(seed)
    g, node_ids = build_graph(edges_path, directed)
    use_log, rec_type = WEIGHT_MODELS[model]
    prop = g.ep["logw"] if use_log else g.ep["w"]

    t0 = time.time()
    sargs = dict(deg_corr=deg_corr, rec=[prop], rec_types=[rec_type])
    state = gt.minimize_nested_blockmodel_dl(g, state=gt.WeightedBlockState, state_args=sargs)
    ent = float(state.entropy())
    levels = [n_blocks_of(s) for s in state.get_levels()]
    levels = [b for b in levels if b > 0]
    blocks = np.asarray(state.get_levels()[0].get_blocks().a, dtype=np.int64)
    np.savez_compressed(os.path.join(out_dir, name + "_partition.npz"),
                        node_ids=node_ids, blocks=blocks)
    return dict(model=model, deg_corr=deg_corr, directed=directed, seed=seed, nested=True,
                entropy=ent, finite=bool(np.isfinite(ent)),
                n_blocks=int(levels[0]) if levels else 0, levels=levels,
                n_vertices=int(g.num_vertices()), n_edges=int(g.num_edges()),
                elapsed_s=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default=os.path.expanduser("~/edges_t5_dir.npz"))
    ap.add_argument("--out", default=os.path.expanduser("~/nested_results"))
    ap.add_argument("--jobs", type=int, default=3, help="parallel cells (each pinned to 1 thread)")
    ap.add_argument("--seeds", default="0", help="comma-separated, e.g. 0,1,2")
    ap.add_argument("--seed", type=int, default=0, help="internal (subprocess)")
    ap.add_argument("--cell", help="internal (subprocess): model,dc,directed,name")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.cell:                                     # child process: run exactly one fit
        model, dc, di, name = a.cell.split(",")
        try:
            r = run_cell(model, dc == "1", di == "1", a.seed, a.edges, a.out, name)
        except Exception as e:
            r = dict(model=model, deg_corr=dc == "1", directed=di == "1",
                     error=f"{type(e).__name__}: {e}")
        with open(os.path.join(a.out, name + ".json"), "w") as f:
            json.dump(r, f, indent=2)
        return

    # seed is the OUTER loop: seed 0 completes the whole grid before seed 1 starts, so an
    # interrupted run still leaves a complete table rather than a ragged one.
    seeds = [int(s) for s in a.seeds.split(",")]
    cells = [(m, dc, di, f"{m}_{'dc' if dc else 'ndc'}_{'dir' if di else 'und'}_s{sd}", sd)
             for sd in seeds for m in PRIORITY for dc in (True, False) for di in (True, False)]
    todo = [c for c in cells if not os.path.exists(os.path.join(a.out, c[3] + ".json"))]
    print(f"[nested] {len(cells)} cells ({len(seeds)} seeds x 16) | {len(cells)-len(todo)} cached "
          f"| {len(todo)} to run | {a.jobs} parallel", flush=True)
    print(f"[nested] edges={a.edges}\n[nested] out={a.out}\n", flush=True)

    t0, running = time.time(), []
    while todo or running:
        while todo and len(running) < a.jobs:
            c = todo.pop(0)
            spec = f"{c[0]},{int(c[1])},{int(c[2])},{c[3]}"
            env = dict(os.environ, OMP_NUM_THREADS="1")
            pr = subprocess.Popen([sys.executable, __file__, "--edges", a.edges, "--out", a.out,
                                   "--seed", str(c[4]), "--cell", spec], env=env)
            running.append((pr, c))
            print(f"[start] {c[3]}  (+{(time.time()-t0)/60:.0f} min)", flush=True)
        time.sleep(10)
        for pr, c in list(running):
            if pr.poll() is None:
                continue
            running.remove((pr, c))
            path = os.path.join(a.out, c[3] + ".json")
            if not os.path.exists(path):
                print(f"[CRASH] {c[3]}  rc={pr.returncode}"
                      f"{' (SEGFAULT)' if pr.returncode in (-11, 139) else ''}", flush=True)
                continue
            r = json.load(open(path))
            if r.get("entropy") is None:
                print(f"[FAIL ] {c[3]}  {r.get('error')}", flush=True)
            else:
                print(f"[done ] {c[3]}  entropy={r['entropy']:.1f}  blocks={r['n_blocks']}  "
                      f"levels={r['levels'][:6]}  {r['elapsed_s']/60:.0f} min", flush=True)
    print(f"\n[nested] finished in {(time.time()-t0)/3600:.2f} h -> {a.out}")


if __name__ == "__main__":
    main()
