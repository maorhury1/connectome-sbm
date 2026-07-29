"""
NESTED SBM sweep — the hierarchy experiment (RQ-D / E4), portable.

Grid: --models x {DC, non-DC} x {directed, undirected} x --seeds, nested, on the canonical >=5
connectome, from the .npz written by export_edges.py. All six weight likelihoods are SUPPORTED
(lognormal included); which ones actually run is chosen at launch with --models, so the
production sweep can simply drop the tags it does not want.

REQUIRES graph-tool >= 3.1. On graph-tool 2.98 the weighted NESTED fit does not work, and 3.1+
cannot run on the lab server (needs glibc 2.38, server has 2.35) — so this script is meant to be
run on a machine that can (macOS, or an Ubuntu 24.04+ host). Verified on graph-tool 3.5 / macOS
arm64: 16/16 nested cells succeed on synthetic data, including nested + lognormal + DC.

Robustness (learned the hard way):
  - every cell runs in its OWN subprocess -> a segfault is recorded, not fatal;
  - every finished cell writes its own JSON + partition .npz -> re-running SKIPS completed cells,
    so an interrupted overnight run resumes for free;
  - seed is the OUTER loop, so an interrupted run still leaves a COMPLETE grid at seed 0.

Usage:
    # production nested sweep (80 fits: 4 models x 2 dc x 2 dir x 5 seeds)
    python nested_sweep.py --edges edges_t5_dir.npz --out nested_results --jobs 4 \
        --models gaussian,poisson,geometric,exponential --seeds 0,1,2,3,4
    # include lognormal explicitly if you want to attempt it
    python nested_sweep.py --models lognormal --seeds 0
    python report_nested.py --out nested_results        # table when done (or partway)
"""
import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np

# ALL available weight likelihoods. Nothing is excluded here on purpose: pick what to run with
# --models at launch time (e.g. lognormal is available but is known not to converge nested on the
# full graph, so the production sweep simply omits it).
WEIGHT_MODELS = {
    "lognormal":   (True,  "real-normal"),        # normal on log(w)
    "gaussian":    (False, "real-normal"),        # normal on raw w
    "poisson":     (False, "discrete-poisson"),
    "geometric":   (False, "discrete-geometric"),
    "exponential": (False, "real-exponential"),
    "binomial":    (False, "discrete-binomial"),
}
DEFAULT_MODELS = ["gaussian", "poisson", "geometric", "exponential"]   # lognormal omitted


def n_blocks_of(state):
    """Block count across graph-tool versions (3.5 dropped get_nonempty_B on WeightedBlockState)."""
    for m in ("get_nonempty_B", "get_B"):
        if hasattr(state, m):
            try:
                return int(getattr(state, m)())
            except Exception:
                pass
    return int(len(np.unique(state.get_blocks().a)))


def fit_nested(gt, g, prop, rec_type, deg_corr):
    """Nested weighted fit.

    CRITICAL API NOTE: the weight likelihood must go to the BASE state, via
    `base_state=` / `base_state_args=`. Passing `state=WeightedBlockState, state_args=...`
    silently replaces the *nested* class instead and the weights are IGNORED — the symptom is
    identical entropies across different weight models (caught by --verify).
    """
    bargs = dict(deg_corr=deg_corr, rec=[prop], rec_types=[rec_type])
    if hasattr(gt, "WeightedBlockState"):                     # graph-tool 3.x
        return gt.minimize_nested_blockmodel_dl(
            g, base_state=gt.WeightedBlockState, base_state_args=bargs)
    # graph-tool 2.x: weights are covariates on BlockState itself
    return gt.minimize_nested_blockmodel_dl(
        g, state_args=dict(deg_corr=deg_corr, recs=[prop], rec_types=[rec_type]))


def verify_weights_applied(edges_path, n_sub=4000):
    """Guard against the silent-unweighted bug.

    Fits every weight model on the SAME dense subgraph (the n_sub highest-degree nodes -- taking
    the first n_sub indices gives an almost edgeless graph and tells you nothing). Distinct
    finite description lengths => the weight likelihood really reaches the nested fit.
    Models returning NaN are reported separately: that is a broken likelihood, not proof that
    weights are ignored.
    """
    import graph_tool.all as gt
    d = np.load(edges_path)
    src, dst, w = d["src"].astype(np.int64), d["dst"].astype(np.int64), d["w"].astype(float)
    deg = np.bincount(np.concatenate([src, dst]))
    top = np.argsort(deg)[::-1][:n_sub]
    remap = -np.ones(deg.size, dtype=np.int64)
    remap[top] = np.arange(len(top))
    keep = (remap[src] >= 0) & (remap[dst] >= 0)
    s2, d2, w2 = remap[src[keep]], remap[dst[keep]], w[keep]
    g = gt.Graph(directed=True); g.add_vertex(len(top))
    g.add_edge_list(np.column_stack([s2, d2]))
    ew = g.new_ep("double"); ew.a = w2
    lw = g.new_ep("double"); lw.a = np.log(w2)
    print(f"[verify] dense subgraph: {g.num_vertices()} nodes / {g.num_edges()} edges", flush=True)

    ents, nan_models = {}, []
    for name, (use_log, rt) in WEIGHT_MODELS.items():
        gt.seed_rng(0); np.random.seed(0)
        try:
            e = float(fit_nested(gt, g, lw if use_log else ew, rt, True).entropy())
        except Exception as ex:
            print(f"[verify]   {name:10} ERROR {type(ex).__name__}: {str(ex)[:60]}", flush=True)
            nan_models.append(name); continue
        if np.isfinite(e):
            ents[name] = e
            print(f"[verify]   {name:10} entropy = {e:14.1f}", flush=True)
        else:
            nan_models.append(name)
            print(f"[verify]   {name:10} entropy = NaN  (likelihood broken for nested)", flush=True)

    vals = sorted(ents.values())
    spread = (vals[-1] - vals[0]) if len(vals) > 1 else 0.0
    ok = len(ents) >= 2 and spread > 1.0
    print(f"[verify] {len(ents)} finite / {len(WEIGHT_MODELS)} models; spread = {spread:.1f} nats")
    if nan_models:
        print(f"[verify] NaN/error models (will be excluded from the sweep): {nan_models}")
    print(f"[verify] -> {'PASS: weights ARE applied' if ok else 'FAIL: fits do not differ by weight model'}",
          flush=True)
    return ok


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
      blockmat.npz  : block-pair edge count / weight sum / weight sum-of-squares AT EVERY
                      SAVED LEVEL (suffixed _0.._L; block-pair weight means+variances are
                      recoverable from these, for the whole hierarchy, without refitting).
      <name>.json   : total MDL, per-level MDL, blocks per level, config, timing.
    """
    import graph_tool.all as gt
    gt.seed_rng(seed); np.random.seed(seed)
    g, node_ids = build_graph(edges_path, directed)
    use_log, rec_type = WEIGHT_MODELS[model]
    prop = g.ep["logw"] if use_log else g.ep["w"]

    t0 = time.time()
    state = fit_nested(gt, g, prop, rec_type, deg_corr)
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

    # --- block-pair stats at EVERY saved level, stored SPARSE (only occupied pairs) ---
    # dense K x K would be gigabytes when K is in the thousands; sparse triplets are tiny and
    # lose nothing: block-pair weight mean/variance are recoverable from ecount/wsum/wsq.
    # Saved per level so weight statistics are available for the whole hierarchy without refit.
    try:
        E = g.get_edges()
        wv = np.asarray(g.ep["w"].a, dtype=float)
        bm = {}
        for key in [k for k in save if k.startswith("level_")]:
            l = key.split("_")[1]
            uniq, inv = np.unique(save[key], return_inverse=True)
            K = len(uniq)
            flat = inv[E[:, 0]].astype(np.int64) * K + inv[E[:, 1]].astype(np.int64)
            pair, idx = np.unique(flat, return_inverse=True)
            bm[f"block_ids_{l}"] = uniq
            bm[f"K_{l}"] = np.int64(K)
            bm[f"row_{l}"] = (pair // K).astype(np.int32)
            bm[f"col_{l}"] = (pair % K).astype(np.int32)
            bm[f"ecount_{l}"] = np.bincount(idx, minlength=len(pair)).astype(np.int64)
            bm[f"wsum_{l}"] = np.bincount(idx, weights=wv, minlength=len(pair))
            bm[f"wsq_{l}"] = np.bincount(idx, weights=wv * wv, minlength=len(pair))
        np.savez_compressed(os.path.join(out_dir, name + "_blockmat.npz"), **bm)
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
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated; any of " + ",".join(WEIGHT_MODELS))
    ap.add_argument("--seeds", default="0", help="comma-separated, e.g. 0,1,2")
    ap.add_argument("--seed", type=int, default=0, help="internal (subprocess)")
    ap.add_argument("--cell", help="internal (subprocess): model,dc,directed,name")
    ap.add_argument("--timeout", type=float, default=12.0,
                    help="hours per fit before it is killed (0 = no timeout)")
    ap.add_argument("--max-retries", type=int, default=2,
                    help="extra attempts with a FRESH seed after a timeout/crash, then give up")
    ap.add_argument("--verify", action="store_true",
                    help="check the weight likelihood actually reaches the nested fit, then exit")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.verify:
        sys.exit(0 if verify_weights_applied(a.edges) else 1)

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
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    bad = [m for m in models if m not in WEIGHT_MODELS]
    if bad:
        sys.exit(f"unknown model(s): {bad}; available: {list(WEIGHT_MODELS)}")
    cells = [(m, dc, di, f"{m}_{'dc' if dc else 'ndc'}_{'dir' if di else 'und'}_s{sd}", sd)
             for sd in seeds for m in models for dc in (True, False) for di in (True, False)]
    todo = [c for c in cells if not os.path.exists(os.path.join(a.out, c[3] + ".json"))]
    print(f"[nested] {len(cells)} cells ({len(models)} models x 2 dc x 2 dir x {len(seeds)} seeds) "
          f"| {len(cells)-len(todo)} cached "
          f"| {len(todo)} to run | {a.jobs} parallel", flush=True)
    print(f"[nested] edges={a.edges}\n[nested] out={a.out}\n", flush=True)

    # Per-fit wall-clock cap + retry-with-a-fresh-seed. Historical full-brain nested fits took
    # 3.7-4.8 h, so the default 12 h is ~2.5x headroom: a legitimately slow fit is never killed,
    # but a stuck one (as lognormal does) frees its worker slot instead of blocking the queue.
    cap = a.timeout * 3600 if a.timeout > 0 else float("inf")
    t0, running, tally = time.time(), [], {"done": 0, "failed": 0, "retried": 0}

    def launch(c, attempt):
        """attempt 0 = original seed/name; retries get a fresh seed and a _retryN name."""
        seed = c[4] if attempt == 0 else c[4] + 1000 * attempt
        name = c[3] if attempt == 0 else f"{c[3]}_retry{attempt}"
        spec = f"{c[0]},{int(c[1])},{int(c[2])},{name}"
        env = dict(os.environ, OMP_NUM_THREADS="1")
        pr = subprocess.Popen([sys.executable, __file__, "--edges", a.edges, "--out", a.out,
                               "--seed", str(seed), "--cell", spec], env=env)
        running.append(dict(pr=pr, cell=c, attempt=attempt, name=name,
                            seed=seed, start=time.time()))
        tag = "" if attempt == 0 else f"  [retry {attempt}/{a.max_retries}, seed {seed}]"
        print(f"[start] {name}  (+{(time.time()-t0)/60:.0f} min){tag}", flush=True)

    def requeue_or_give_up(j, why):
        """Retry with a new seed, or record permanent failure and move on."""
        if j["attempt"] < a.max_retries:
            tally["retried"] += 1
            print(f"[{why:6}] {j['name']}  -> retrying with a fresh seed", flush=True)
            launch(j["cell"], j["attempt"] + 1)
        else:
            tally["failed"] += 1
            print(f"[GIVEUP] {j['cell'][3]}  after {a.max_retries + 1} attempts ({why})", flush=True)
            with open(os.path.join(a.out, j["cell"][3] + ".json"), "w") as fh:
                json.dump(dict(model=j["cell"][0], deg_corr=j["cell"][1], directed=j["cell"][2],
                               seed=j["cell"][4], nested=True, status="FAILED", reason=why,
                               attempts=a.max_retries + 1), fh, indent=2)

    while todo or running:
        while todo and len(running) < a.jobs:
            launch(todo.pop(0), 0)
        time.sleep(10)
        for j in list(running):
            elapsed = time.time() - j["start"]
            if j["pr"].poll() is None:
                if elapsed > cap:                       # hung: kill and free the slot
                    j["pr"].kill()
                    running.remove(j)
                    print(f"[TIMEOUT] {j['name']} after {elapsed/3600:.1f} h", flush=True)
                    requeue_or_give_up(j, "TIMEOUT")
                continue
            running.remove(j)
            path = os.path.join(a.out, j["name"] + ".json")
            if not os.path.exists(path):
                rc = j["pr"].returncode
                print(f"[CRASH ] {j['name']}  rc={rc}"
                      f"{' (SEGFAULT)' if rc in (-11, 139) else ''}", flush=True)
                requeue_or_give_up(j, "CRASH")
                continue
            r = json.load(open(path))
            if r.get("entropy") is None:
                print(f"[FAIL  ] {j['name']}  {r.get('error')}", flush=True)
                requeue_or_give_up(j, "ERROR")
            else:
                tally["done"] += 1
                print(f"[done  ] {j['name']}  entropy={r['entropy']:.1f}  blocks={r['n_blocks']}  "
                      f"levels={r['levels'][:6]}  {r['elapsed_s']/60:.0f} min  "
                      f"[{tally['done']}/{len(cells)}]", flush=True)
    print(f"\n[nested] finished in {(time.time()-t0)/3600:.2f} h -> {a.out}")
    print(f"[nested] {tally['done']} ok, {tally['failed']} failed, "
          f"{tally['retried']} retries used", flush=True)


if __name__ == "__main__":
    main()
