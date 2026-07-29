"""
Validation gate for CPWDL (spec Sec. 11 + Sec. 16). NOTHING may be interpreted on real data
until this passes.

  A. normalisation   : every conditioned pmf sums to 1 over its integer support; log-probs
                       finite on observed weights; no arbitrary floor.
  B. sampling        : simulated frequencies match the implemented probabilities.
  C. recovery (dist) : synthetic samples from family F, scored by all five -> F should win.
  D. recovery (bundle): fixed partition, real bundle sizes, synthetic weights from F -> F wins.
  E. tiny-graph      : end-to-end CPWDL on a small synthetic graph runs and is finite.

Predeclared pass rule (Sec. 11.4): >= 80% correct selection in C at n >= 200, counting a tie
with a near-equivalent family (|gap| < 2 nats) as correct. Exponential/geometric are formally
near-equivalent (continuous vs discrete twins) and are treated as one class for scoring.

Run:  python validate_candidate_mdl.py
"""
import json
import numpy as np
import integer_weight_models as IWM
import bundle_description as BD

WMIN = IWM.WMIN_DEFAULT
NEAR = {frozenset(("exponential", "geometric"))}      # formally near-equivalent pair
REPLICATES, NSIZES = 12, (50, 200, 1000)
PASS_RATE = 0.80

PARAM_GRID = {                                        # weak / moderate / heavy dispersion
    "poisson":     [{"lam": 8.0}, {"lam": 20.0}, {"lam": 60.0}],
    "geometric":   [{"p": 0.30}, {"p": 0.12}, {"p": 0.04}],
    "exponential": [{"scale": 4.0}, {"scale": 12.0}, {"scale": 40.0}],
    "gaussian":    [{"mu": 10.0, "sigma": 2.0}, {"mu": 20.0, "sigma": 8.0},
                    {"mu": 40.0, "sigma": 20.0}],
    "lognormal":   [{"mu": 2.2, "sigma": 0.4}, {"mu": 2.5, "sigma": 0.8},
                    {"mu": 3.0, "sigma": 1.2}],
}


def A_normalisation():
    rows, ok_all = [], True
    for fam in IWM.FAMILIES:
        for p in PARAM_GRID[fam]:
            m = IWM.total_mass(fam, p, WMIN)
            below = np.exp(IWM.logpmf(fam, np.arange(0, WMIN), p, WMIN)).sum()
            lp = IWM.logpmf(fam, np.array([WMIN, 10, 50, 500, 2633]), p, WMIN)
            ok = abs(m - 1) < 1e-6 and below == 0.0 and np.all(np.isfinite(lp))
            ok_all &= ok
            rows.append(dict(test="normalisation", family=fam, params=str(p),
                             mass=round(m, 9), mass_below_wmin=float(below),
                             finite_logp=bool(np.all(np.isfinite(lp))), ok=bool(ok)))
    return ok_all, rows


def B_sampling(n=200000):
    rng = np.random.default_rng(0)
    rows, ok_all = [], True
    for fam in IWM.FAMILIES:
        p = PARAM_GRID[fam][1]
        s = IWM.rvs(fam, p, n, WMIN, rng)
        ks, cnt = np.unique(s, return_counts=True)
        emp = cnt / n
        th = np.exp(IWM.logpmf(fam, ks, p, WMIN))
        keep = th > 1e-4                              # only bins with enough samples
        err = float(np.max(np.abs(emp[keep] - th[keep]))) if keep.any() else 0.0
        ok = err < 0.01
        ok_all &= ok
        rows.append(dict(test="sampling", family=fam, max_abs_err=round(err, 5), ok=bool(ok)))
    return ok_all, rows


def _score_all(w):
    """CPWDL weight+parameter terms for one sample under every family (topology cancels)."""
    out = {}
    for fam in IWM.FAMILIES:
        p, info = IWM.fit(fam, w, WMIN)
        if not info.get("ok"):
            out[fam] = np.inf; continue
        lp = IWM.logpmf(fam, w, p, WMIN)
        if not np.all(np.isfinite(lp)):
            out[fam] = np.inf; continue
        out[fam] = -float(lp.sum()) + 0.5 * IWM.N_PARAMS[fam] * np.log(len(w))
    return out


def _correct(true_fam, scores):
    best = min(scores, key=scores.get)
    if best == true_fam:
        return True
    return (frozenset((best, true_fam)) in NEAR
            and abs(scores[best] - scores[true_fam]) < 2.0)


def C_recovery_distribution():
    rng = np.random.default_rng(1)
    rows = []
    for fam in IWM.FAMILIES:
        for p in PARAM_GRID[fam]:
            for n in NSIZES:
                hits = 0
                for r in range(REPLICATES):
                    w = IWM.rvs(fam, p, n, WMIN, np.random.default_rng(rng.integers(1 << 31)))
                    hits += _correct(fam, _score_all(w))
                rows.append(dict(test="recovery_dist", family=fam, params=str(p), n=n,
                                 rate=hits / REPLICATES))
    big = [r for r in rows if r["n"] >= 200]
    rate = float(np.mean([r["rate"] for r in big]))
    return rate >= PASS_RATE, rows, rate


def D_recovery_bundles(n_bundles=60, seed=2):
    """Realistic bundle-size mixture; scored through the real grouping/pooling path."""
    rng = np.random.default_rng(seed)
    sizes = np.maximum(rng.lognormal(3.0, 1.2, n_bundles).astype(int), 2)
    rows = []
    for fam in IWM.FAMILIES:
        p = PARAM_GRID[fam][1]
        segs = [IWM.rvs(fam, p, int(m), WMIN, rng) for m in sizes]
        w = np.concatenate(segs)
        starts = np.cumsum([0] + [len(s) for s in segs])[:-1]
        ends = np.cumsum([len(s) for s in segs])
        local, pooled = BD.group_bundles(w, starts, ends)
        sc = {}
        for f2 in IWM.FAMILIES:
            r = BD.score_groups(f2, local, pooled, WMIN)
            sc[f2] = (r["weight_nll"] + r["parameter_dl"]) if r["ok"] else np.inf
        rows.append(dict(test="recovery_bundle", family=fam, n_edges=int(len(w)),
                         n_local=len(local), n_pooled=int(len(pooled)),
                         winner=min(sc, key=sc.get), correct=bool(_correct(fam, sc)),
                         **{f"dl_{k}": round(v, 1) for k, v in sc.items()}))
    return all(r["correct"] for r in rows), rows


def E_tiny_graph():
    """End-to-end on a small synthetic graph: topology term + weight term, finite."""
    try:
        import graph_tool.all as gt
        import topology_description as TD
    except Exception as e:
        return False, [dict(test="tiny_graph", ok=False, reason=str(e)[:80])]
    rng = np.random.default_rng(3)
    n, nb = 300, 5
    blocks = rng.integers(0, nb, n)
    src = rng.integers(0, n, 4000); dst = rng.integers(0, n, 4000)
    keep = src != dst; src, dst = src[keep], dst[keep]
    g = gt.Graph(directed=True); g.add_vertex(n)
    g.add_edge_list(np.column_stack([src, dst]))
    w = IWM.rvs("lognormal", {"mu": 2.5, "sigma": 0.8}, g.num_edges(), WMIN, rng)
    S_A = TD.topology_dl(g, blocks)
    starts, ends, order = BD.make_bundles(blocks[src], blocks[dst], w)
    local, pooled = BD.group_bundles(w[order], starts, ends)
    r = BD.score_groups("lognormal", local, pooled, WMIN)
    total = S_A + r["weight_nll"] + r["parameter_dl"] + np.log(5)
    ok = r["ok"] and np.isfinite(S_A) and np.isfinite(total)
    return ok, [dict(test="tiny_graph", S_A=round(S_A, 1),
                     weight_nll=round(r["weight_nll"], 1),
                     parameter_dl=round(r["parameter_dl"], 1),
                     total=round(total, 1), ok=bool(ok))]


def main():
    import pandas as pd
    allrows, results = [], {}
    print("CPWDL validation gate\n" + "=" * 70, flush=True)

    for name, fn in (("A normalisation", A_normalisation), ("B sampling", B_sampling)):
        ok, rows = fn(); results[name] = ok; allrows += rows
        print(f"{name:22} {'PASS' if ok else 'FAIL'}", flush=True)

    ok, rows, rate = C_recovery_distribution()
    results["C recovery (dist)"] = ok; allrows += rows
    print(f"{'C recovery (dist)':22} {'PASS' if ok else 'FAIL'}  "
          f"(mean rate at n>=200: {rate:.2f}, need {PASS_RATE})", flush=True)

    ok, rows = D_recovery_bundles(); results["D recovery (bundle)"] = ok; allrows += rows
    print(f"{'D recovery (bundle)':22} {'PASS' if ok else 'FAIL'}", flush=True)

    ok, rows = E_tiny_graph(); results["E tiny graph"] = ok; allrows += rows
    print(f"{'E tiny graph':22} {'PASS' if ok else 'FAIL'}", flush=True)

    import os, config
    out = config.REPO_DIR / "artifacts" / "candidate_mdl"
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(allrows).to_csv(out / "synthetic_validation.csv", index=False)

    print("=" * 70)
    print("\nper-family recovery at n>=200 (C):")
    d = pd.DataFrame([r for r in allrows if r["test"] == "recovery_dist" and r["n"] >= 200])
    print(d.groupby("family")["rate"].mean().round(2).to_string())
    print("\nbundle-level recovery (D):")
    b = pd.DataFrame([r for r in allrows if r["test"] == "recovery_bundle"])
    print(b[["family", "winner", "correct", "n_edges", "n_local", "n_pooled"]].to_string(index=False))

    gate = all(results.values())
    print(f"\nGATE: {'PASS -- scorer may be used on real data' if gate else 'FAIL -- do NOT run on the connectome'}")
    json.dump({k: bool(v) for k, v in results.items()},
              open(out / "validation_gate.json", "w"), indent=2)
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
