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
    """One nested fit. Saves EVERYTHING needed for later analysis without refitting:

      partition.npz : level_0 .. level_L  -> each neuron's block AT EVERY LEVEL of the
                      hierarchy (projected onto the original nodes; this is what the
                      hierarchy/granularity analysis needs), plus node_ids and the raw
                      per-level block vectors (`bs_*`) describing the tree itself.
      blockmat.npz  : finest-level block-pair edge count / weight sum / weight sum-of-squares
                      (block-pair weight means+variances are recoverable from these).
      <name>.json   : total MDL, per-level MDL, blocks per level, config, timing.
    """
    import graph_tool.all as gt
    gt.seed_rng(seed); np.random.seed(seed)
    g, node_ids = build_graph(edges_path, directed)
    use_log, rec_type = WEIGHT_MODELS[model]
    prop = g.ep["logw"] if use_log else g.ep["w"]

    t0 = time.time()
    sargs = dict(deg_corr=deg_corr, rec=[prop], rec_types=[rec_type])
    state = gt.minimize_nested_blockmodel_dl(g, state=gt.WeightedBlockState, state_args=sargs)
    ent = float(state.entropy())
    elapsed = time.time() - t0

    lvl_states = state.get_levels()
    levels = [n_blocks_of(s) for s in lvl_states]
    n_real = max(1, len([b for b in levels if b > 1]))     # levels above the collapsed tail

    # --- per-level partition of the ORIGINAL neurons (the hierarchy itself) ---
    save = {"node_ids": node_ids}
    for l in range(len(lvl_states)):
        try:
            b = np.asarray(state.project_level(l).get_blocks().a, dtype=np.int64)
        except Exception:
            if l > 0:
                break
            b = np.asarray(lvl_states[0].get_blocks().a, dtype=np.int64)
        save[f"level_{l}"] = b
        if l >= n_real and len(np.unique(b)) <= 1:
            break                                          # stop once the tree has collapsed
    try:                                                   # raw tree (block-of-block vectors)
        for l, bs in enumerate(state.get_bs()):
            save[f"bs_{l}"] = np.asarray(bs, dtype=np.int64)
    except Exception:
        pass
    np.savez_compressed(os.path.join(out_dir, name + "_partition.npz"), **save)

    # --- finest-level block-pair stats, stored SPARSE (only occupied pairs) ---
    # dense K x K would be gigabytes when K is in the thousands; sparse triplets are tiny and
    # lose nothing: block-pair weight mean/variance are recoverable from ecount/wsum/wsq.
    try:
        fb = save["level_0"]
        uniq, inv = np.unique(fb, return_inverse=True)
        K = len(uniq)
        E = g.get_edges()
        w = np.asarray(g.ep["w"].a, dtype=float)
        flat = inv[E[:, 0]].astype(np.int64) * K + inv[E[:, 1]].astype(np.int64)
        pair, idx = np.unique(flat, return_inverse=True)
        np.savez_compressed(
            os.path.join(out_dir, name + "_blockmat.npz"),
            block_ids=uniq, K=np.int64(K),
            row=(pair // K).astype(np.int32), col=(pair % K).astype(np.int32),
            ecount=np.bincount(idx, minlength=len(pair)).astype(np.int64),
            wsum=np.bincount(idx, weights=w, minlength=len(pair)),
            wsq=np.bincount(idx, weights=w * w, minlength=len(pair)))
    except Exception as e:
        print(f"[warn] blockmat failed for {name}: {e}", flush=True)

    # --- per-level description length ---
    per_level_mdl = []
    for s in lvl_states:
        try:
            per_level_mdl.append(float(s.entropy()))
        except Exception:
            per_level_mdl.append(None)

    sizes = np.bincount(np.unique(save["level_0"], return_inverse=True)[1])
    import platform
    import graph_tool
    return dict(gt_version=str(graph_tool.__version__), platform=platform.platform(),
                edges_file=os.path.basename(edges_path),
                model=model, deg_corr=deg_corr, directed=directed, seed=seed, nested=True,
                entropy=ent, finite=bool(np.isfinite(ent)),
                per_level_mdl=per_level_mdl,
                n_blocks=int(levels[0]) if levels else 0,
                levels=levels, n_levels_nontrivial=n_real,
                block_sizes=dict(n=int(len(sizes)), min=int(sizes.min()), max=int(sizes.max()),
                                 median=float(np.median(sizes)),
                                 n_singletons=int((sizes == 1).sum())),
                n_vertices=int(g.num_vertices()), n_edges=int(g.num_edges()),
                saved_levels=sum(1 for k in save if k.startswith("level_")),
                elapsed_s=round(elapsed, 1))


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
