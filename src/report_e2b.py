"""
E2b report (CP-4, gated): does the label-free predictive winner equal the biological winner?

Reads the cached per-cell JSONs, aggregates held-out predictive log-score (nats per held-out
edge; HIGHER = better prediction) over folds x seeds, and puts it next to the biology column
(V vs primary_type) from the CP-3 cache. Writes the table to RESULTS.md and stops.

Run from src/:  python report_e2b.py
"""
import json
import numpy as np
import pandas as pd
import config

OUT = config.WORK_DIR / "e2b"
SCORES_CACHE = config.WORK_DIR / "eval_scores.csv"


def main():
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    if not rows:
        print("[e2b] no results yet."); return
    df = pd.DataFrame(rows)
    df["dir"] = np.where(df.directed, "dir", "und")
    df["dc"] = np.where(df.deg_corr, "dc", "ndc")

    g = (df.groupby(["model", "dir", "dc", "method"])
           .agg(nats_per_edge=("logscore_per_edge", "mean"),
                sd=("logscore_per_edge", "std"),
                n_cells=("logscore_per_edge", "size"),
                blocks=("n_blocks", "mean"))
           .reset_index().sort_values("nats_per_edge", ascending=False))

    # biology column from CP-3 (same directed/dc setting)
    if SCORES_CACHE.exists():
        bio = (pd.read_csv(SCORES_CACHE).groupby(["model", "dir", "dc"])["v_primary_type"]
               .mean().reset_index().rename(columns={"v_primary_type": "V_type"}))
        g = g.merge(bio, on=["model", "dir", "dc"], how="left")

    pd.set_option("display.width", 220)
    print("E2b — held-out predictive score (nats/edge, HIGHER = better prediction):\n")
    print(g.round(4).to_string(index=False))

    md = ["\n\n## CP-4 — E2b held-out predictive selection\n",
          f"*Leak-free edge-removed weight prediction, {df.test_frac.iloc[0]:.0%} held out, "
          "3 disjoint folds x 3 seeds, folds shared across models (paired). "
          "nats/edge: HIGHER = better prediction. V(type) from CP-3.*\n",
          "| model | dir | dc | fold method | nats/edge | sd | cells | blocks | V type |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in g.itertuples(index=False):
        v = f"{r.V_type:.3f}" if hasattr(r, "V_type") and pd.notna(r.V_type) else "-"
        md.append(f"| {r.model} | {r.dir} | {r.dc} | {r.method} | **{r.nats_per_edge:.4f}** | "
                  f"{r.sd:.4f} | {r.n_cells} | {r.blocks:.0f} | {v} |")

    # headline: predictive winner vs biology winner, within each setting
    for (d, dc, meth), s in g.groupby(["dir", "dc", "method"]):
        if s.empty or s.V_type.isna().all():
            continue
        pw = s.loc[s.nats_per_edge.idxmax(), "model"]
        bw = s.loc[s.V_type.idxmax(), "model"]
        verdict = "AGREE" if pw == bw else "DISAGREE"
        md.append(f"\n- **{d}/{dc}/{meth}:** prediction picks **{pw}**, biology picks **{bw}** "
                  f"-> **{verdict}**")
        print(f"{d}/{dc}/{meth}: predict -> {pw} | biology -> {bw}  [{verdict}]")

    rp = config.REPO_DIR / "RESULTS.md"
    marker = "\n\n## CP-4 — E2b held-out predictive selection"
    txt = rp.read_text()
    if marker in txt:
        txt = txt[:txt.index(marker)]
    rp.write_text(txt + "\n".join(md) + "\n")
    print("\n[gated] wrote E2b table to RESULTS.md. STOP.")


if __name__ == "__main__":
    main()
