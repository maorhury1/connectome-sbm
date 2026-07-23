"""
Jacobian audit (PLAN Sec 3.5) — make the lognormal-vs-Gaussian MDL comparison legitimate.

"lognormal" is real-normal on log(w); "gaussian" is real-normal on raw w. They describe
DIFFERENT variables, so their description lengths are not comparable as reported. Changing
variables adds a Jacobian term:

    DL_on_w  =  DL_on_log_w  +  sum_e log(w_e)

Step 1 AUDIT (not just applying the formula): the corrected gap between the two models must
be INVARIANT when all weights are rescaled (w -> c*w), because a valid comparison of two
densities of the same quantity cannot depend on the unit. That invariance holds only if
graph-tool's real-normal term behaves like a proper density code. If the gap drifts with c,
the naive correction is wrong and the comparison stays off-limits.

Step 2 APPLY (only if the audit passes): add the Jacobian to the cached lognormal MDLs
(directed and undirected graphs have different edge sets, so each gets its own constant)
and report the corrected lognormal-vs-Gaussian comparison.

Gated: runs, writes RESULTS.md, stops.  Run from src/:  python jacobian_audit.py
"""
import numpy as np
import pandas as pd
import graph_tool.all as gt
import config
import data
import graph as graphmod

SCORES_CACHE = config.WORK_DIR / "eval_scores.csv"


def small_graph(n_nodes=1200):
    """A real subgraph (densest nodes) — faithful weights, small enough to be quick."""
    pre, post, w = data.load_edges(threshold=5, directed=True)
    vals, counts = np.unique(np.concatenate([pre, post]), return_counts=True)
    top = vals[np.argsort(counts)[::-1][:n_nodes]]
    m = np.isin(pre, top) & np.isin(post, top)
    g, _, _ = graphmod.build_graph(pre[m], post[m], w[m], directed=True)
    return g


def dl_pair(g, wvals, b):
    """DL of real-normal on log(w) and on w, for the SAME fixed partition b."""
    lw = g.new_ep("double"); lw.a = np.log(wvals)
    rw = g.new_ep("double"); rw.a = wvals
    s_log = gt.BlockState(g, b=b, recs=[lw], rec_types=["real-normal"], deg_corr=True).entropy()
    s_gau = gt.BlockState(g, b=b, recs=[rw], rec_types=["real-normal"], deg_corr=True).entropy()
    return float(s_log), float(s_gau)


def audit(scales=(1.0, 2.0, 10.0, 100.0), seed=0):
    """Corrected gap must not move when weights are rescaled."""
    g = small_graph()
    w0 = np.asarray(g.ep["w"].a, dtype=float)
    rng = np.random.default_rng(seed)
    b = g.new_vp("int"); b.a = rng.integers(0, 12, g.num_vertices())   # fixed arbitrary partition
    print(f"[audit] subgraph: {g.num_vertices()} nodes, {g.num_edges()} edges, fixed 12-block partition\n")
    print(f"{'scale c':>8} {'DL(log w)':>14} {'DL(w)':>14} {'J=sum log w':>14} {'corrected gap':>15}")
    gaps = []
    for c in scales:
        w = w0 * c
        s_log, s_gau = dl_pair(g, w, b)
        J = float(np.log(w).sum())
        gap = s_log + J - s_gau                      # corrected: both now describe w
        gaps.append(gap)
        print(f"{c:8.1f} {s_log:14.1f} {s_gau:14.1f} {J:14.1f} {gap:15.2f}")
    spread = max(gaps) - min(gaps)
    scale_ref = abs(np.mean(gaps)) + 1e-9
    ok = spread < 0.01 * scale_ref or spread < 1.0
    print(f"\n[audit] corrected-gap spread across scales = {spread:.3f} nats "
          f"({100*spread/scale_ref:.4f}% of the gap)  -> {'PASS' if ok else 'FAIL'}")
    return ok, gaps, spread


def jacobians():
    """sum log(w) for each graph the fits actually used (directed and undirected differ)."""
    out = {}
    for directed, tag in [(True, "dir"), (False, "und")]:
        _, _, w = data.load_edges(threshold=5, directed=directed)
        out[tag] = float(np.log(w.astype(float)).sum())
        print(f"[jacobian] {tag}: {len(w):,} edges, J = sum log w = {out[tag]:,.0f} nats", flush=True)
    return out


def main():
    ok, gaps, spread = audit()
    md = ["\n\n## Jacobian audit — lognormal vs Gaussian MDL\n",
          "*Correction: `DL_on_w = DL_on_log_w + sum_e log(w_e)`. Audit = the corrected gap must "
          "be invariant to rescaling the weights.*\n",
          f"- Audit spread across scales x1..x100: **{spread:.3f} nats** -> "
          f"**{'PASS' if ok else 'FAIL'}**\n"]
    if not ok:
        md.append("\n**Verdict: correction NOT validated — lognormal-vs-Gaussian MDL stays off-limits.**\n")
        (config.REPO_DIR / "RESULTS.md").open("a").write("\n".join(md) + "\n")
        print("\n[gated] audit FAILED; comparison remains invalid. STOP.")
        return

    J = jacobians()
    df = pd.read_csv(SCORES_CACHE)
    df["mdl_corrected"] = df.apply(
        lambda r: r.mdl + J[r["dir"]] if r.model == "lognormal" else r.mdl, axis=1)
    sub = df[df.model.isin(["lognormal", "gaussian"])]
    g = (sub.groupby(["model", "dir", "dc"])
            .agg(mdl_raw=("mdl", "mean"), mdl_corr=("mdl_corrected", "mean"),
                 mdl_corr_sd=("mdl_corrected", "std"), v=("v_primary_type", "mean"))
            .reset_index())
    g[["mdl_raw", "mdl_corr", "mdl_corr_sd"]] /= 1e6
    print("\nCorrected comparison (M nats; lower = better compression):\n")
    print(g.round(3).to_string(index=False))

    md.append("\n| model | dir | dc | MDL raw (M) | **MDL corrected (M)** | V type |")
    md.append("|---|---|---|---|---|---|")
    for r in g.itertuples(index=False):
        md.append(f"| {r.model} | {r.dir} | {r.dc} | {r.mdl_raw:.2f} | "
                  f"**{r.mdl_corr:.2f}±{r.mdl_corr_sd:.2f}** | {r.v:.3f} |")
    for d in ("dir", "und"):
        for dc in ("dc", "ndc"):
            s = g[(g["dir"] == d) & (g.dc == dc)]
            if set(s.model) == {"lognormal", "gaussian"}:
                ln = s[s.model == "lognormal"].iloc[0]
                gs = s[s.model == "gaussian"].iloc[0]
                win = "lognormal" if ln.mdl_corr < gs.mdl_corr else "gaussian"
                agree = "AGREE" if (win == "lognormal") == (ln.v > gs.v) else "DISAGREE"
                md.append(f"\n- **{d}/{dc}:** MDL picks **{win}** "
                          f"({ln.mdl_corr:.2f} vs {gs.mdl_corr:.2f} M); biology (V) picks "
                          f"**{'lognormal' if ln.v > gs.v else 'gaussian'}** -> **{agree}**")
                print(f"{d}/{dc}: MDL -> {win}; V -> {'lognormal' if ln.v > gs.v else 'gaussian'}  [{agree}]")
    (config.REPO_DIR / "RESULTS.md").open("a").write("\n".join(md) + "\n")
    print("\n[gated] wrote Jacobian audit + corrected comparison to RESULTS.md. STOP.")


if __name__ == "__main__":
    main()
