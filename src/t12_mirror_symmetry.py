"""t12: does the recovered retinotopic map respect the mirror symmetry of the two optic lobes?

The two optic lobes are mirror images.  A weight model that genuinely recovers a
retinotopic map (rather than merely spatially-blobby blocks) should have left-hemisphere
blocks that correspond one-to-one with right-hemisphere blocks under a *specific*
geometric transform of the hexagonal (p,q) eye coordinates.  That is a falsifiable
prediction: the wrong transform, or no transform, should fail.

Pipeline
  1. per (model, seed): block centroids in (p,q), computed separately per hemisphere,
     restricted to neurons that carry a published column assignment.
  2. candidate transforms applied to the RIGHT hemisphere centroids:
       identity (p,q), swap (q,p), (-p,q), (p,-q), (-p,-q), (-q,-p), (q,-p), (-q,p)
     Convention in this project: DV = p+q, AP = p-q.  So (q,p) preserves DV and negates
     AP (a true reflection of the hex lattice across the DV axis); (-q,-p) preserves AP
     and negates DV; (-p,-q) is a 180-degree rotation; (-p,q) / (p,-q) are not lattice
     isometries (they swap+shear the DV/AP axes) but are tested as requested.
  3. Hungarian matching of left <-> transformed-right centroids, mean residual distance.
  4. Nulls (>= 20 draws): (a) shuffle block identities within the right hemisphere,
     preserving block sizes; (b) random one-to-one pairing of the real centroids.

Also: a *convention audit*.  If the published column file already stores the two
hemispheres in a mirrored (eye-centric) frame, the identity transform is the anatomical
mirror and wins for a boring reason.  We settle that independently of the SBM by
regressing 3D neuron centroids (FlyWire nm) on (p,q) per hemisphere: the dorsoventral
and anterior-posterior axes are mirror-invariant, the medio-lateral axis flips sign.

Run:  cd src && $ENV_PYTHON t12_mirror_symmetry.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import config  # noqa: F401  (kept for path provenance / consistency with the rest of src)
import data

RESULTS = Path("/var/tmp/csbm_work/results")
COLFILE = Path("/var/tmp/csbm_work/scratch/column_assignment.csv.gz")
CENTROIDS = Path("/var/tmp/csbm_work/scratch/centroids.parquet")

MODELS = ["lognormal", "gaussian", "geometric", "exponential", "poisson"]
SEEDS = [0, 1, 2, 3, 4]
TAG = "t5_dir_dc"

MIN_BLOCK = 3      # min neurons-with-coordinates a block needs in a hemisphere to get a centroid
N_NULL = 25        # null draws per run
RNG = np.random.default_rng(0)

TRANSFORMS = {
    "identity (p,q)":  lambda p, q: (p, q),
    "swap (q,p)":      lambda p, q: (q, p),
    "negp (-p,q)":     lambda p, q: (-p, q),
    "negq (p,-q)":     lambda p, q: (p, -q),
    "negboth (-p,-q)": lambda p, q: (-p, -q),
    "swapneg (-q,-p)": lambda p, q: (-q, -p),
    "rot (q,-p)":      lambda p, q: (q, -p),
    "rot (-q,p)":      lambda p, q: (-q, p),
}


# ----------------------------------------------------------------------------------
# convention audit: what does (p,q) mean anatomically in each hemisphere?
# ----------------------------------------------------------------------------------
def convention_audit(col: pd.DataFrame) -> None:
    cen = pd.read_parquet(CENTROIDS).rename(columns={"x": "X", "y": "Y", "z": "Z"})
    m = col.join(cen[["X", "Y", "Z"]], on="root_id", how="inner")
    print("\n" + "=" * 88)
    print("COORDINATE-CONVENTION AUDIT  (3D FlyWire centroids ~ p,q, per hemisphere)")
    print("FlyWire axes: X = medio-lateral (flips under a mirror), Y = dorsoventral,")
    print("              Z = anterior-posterior (Y and Z are mirror-INVARIANT).")
    print("=" * 88)
    maps = {}
    for ct in ["Mi1", "T4a", "L1", "<all columnar>"]:
        sub = m if ct == "<all columnar>" else m[m["type"] == ct]
        print(f"\n  cell type {ct!r}  (n={len(sub)})")
        for h, g in sub.groupby("hemisphere"):
            A = np.c_[g.p.values, g.q.values, np.ones(len(g))]
            row = {}
            for t in ["X", "Y", "Z"]:
                coef, *_ = np.linalg.lstsq(A, g[t].values, rcond=None)
                r = np.corrcoef(A @ coef, g[t].values)[0, 1]
                row[t] = (coef[0], coef[1], r)
            print(f"    {h:>5}  n={len(g):5d}  "
                  + "  ".join(f"d{t}/dp={row[t][0]:8.0f} d{t}/dq={row[t][1]:8.0f} (R={row[t][2]:.3f})"
                              for t in ["X", "Y", "Z"]))
            if ct == "Mi1":
                maps[h] = np.array([[row["Y"][0], row["Y"][1]],
                                    [row["Z"][0], row["Z"][1]]])
    # transform taking right-(p,q) to the left-(p,q) that sits at the mirror-image
    # anatomical location (same Y,Z):   M_L @ v_L = M_R @ v_R  =>  v_L = M_L^-1 M_R v_R
    A = np.linalg.solve(maps["left"], maps["right"])
    print("\n  Implied map from RIGHT (p,q) to the anatomically mirror-homologous LEFT (p,q)")
    print("  (derived from Mi1, which is one cell per column):")
    print("     [[%.3f, %.3f],\n      [%.3f, %.3f]]" % (A[0, 0], A[0, 1], A[1, 0], A[1, 1]))
    print("     distance from identity  ||A - I||_F = %.3f" % np.linalg.norm(A - np.eye(2)))
    print("     distance from swap      ||A - S||_F = %.3f"
          % np.linalg.norm(A - np.array([[0., 1.], [1., 0.]])))
    verdict = "IDENTITY (file stores the hemispheres in a MIRRORED / eye-centric frame)" \
        if np.linalg.norm(A - np.eye(2)) < np.linalg.norm(A - np.array([[0., 1.], [1., 0.]])) \
        else "SWAP (file stores both hemispheres in a common frame; a reflection is required)"
    print(f"  => anatomically correct left<->right correspondence is: {verdict}")


# ----------------------------------------------------------------------------------
# core scoring
# ----------------------------------------------------------------------------------
def centroids(blocks: np.ndarray, pq: np.ndarray, min_n: int):
    """Return (centroid array [k,2], block ids, sizes) for one hemisphere."""
    order = np.argsort(blocks, kind="stable")
    b = blocks[order]
    v = pq[order]
    edges = np.flatnonzero(np.r_[True, b[1:] != b[:-1], True])
    ids, cs, ns = [], [], []
    for i, j in zip(edges[:-1], edges[1:]):
        if j - i >= min_n:
            ids.append(b[i])
            cs.append(v[i:j].mean(axis=0))
            ns.append(j - i)
    if not cs:
        return np.zeros((0, 2)), np.array([]), np.array([])
    return np.array(cs), np.array(ids), np.array(ns)


def matched_stats(CL: np.ndarray, CR: np.ndarray, center: bool) -> tuple[float, float]:
    """(mean residual, fraction of matched pairs within 1 hex unit) after Hungarian matching.

    The mean is near-degenerate among transforms that map the (p,q) point cloud onto itself
    (the eye field is roughly centrally symmetric), so `frac<1` -- essentially-exact column
    correspondences -- is the sharper statistic.
    """
    if len(CL) == 0 or len(CR) == 0:
        return np.nan, np.nan
    if center:
        CR = CR - CR.mean(0) + CL.mean(0)
    D = cdist(CL, CR)
    r, c = linear_sum_assignment(D)
    d = D[r, c]
    return float(d.mean()), float((d < 1.0).mean())


def matched_residual(CL, CR, center):
    return matched_stats(CL, CR, center)[0]


def all_transform_residuals(CL, CR, center=True, want="mean") -> dict:
    out = {}
    for name, f in TRANSFORMS.items():
        p, q = f(CR[:, 0], CR[:, 1])
        mu, fr = matched_stats(CL, np.c_[p, q], center)
        out[name] = mu if want == "mean" else fr
    return out


def random_pair_residual(CL, CR, rng, center=True) -> float:
    if center:
        CR = CR - CR.mean(0) + CL.mean(0)
    k = min(len(CL), len(CR))
    il = rng.permutation(len(CL))[:k]
    ir = rng.permutation(len(CR))[:k]
    return float(np.linalg.norm(CL[il] - CR[ir], axis=1).mean())


def run_one(model: str, seed: int, col: pd.DataFrame, side: pd.Series) -> dict | None:
    f = RESULTS / f"{model}_{TAG}_s{seed}" / "partition.npz"
    if not f.exists():
        print(f"  [skip] missing {f}")
        return None
    z = np.load(f)
    part = pd.DataFrame({"root_id": z["node_ids"], "block": z["blocks"]})
    df = part.merge(col, on="root_id", how="inner")

    nb_total = len(np.unique(z["blocks"]))
    hemi = df["hemisphere"].values
    L = df[hemi == "left"]
    R = df[hemi == "right"]

    CL, idL, nL = centroids(L["block"].values, L[["p", "q"]].values.astype(float), MIN_BLOCK)
    CR, idR, nR = centroids(R["block"].values, R[["p", "q"]].values.astype(float), MIN_BLOCK)
    bilateral = len(np.intersect1d(idL, idR))

    obs = all_transform_residuals(CL, CR, center=True)
    obs_raw = all_transform_residuals(CL, CR, center=False)
    obs_fr = all_transform_residuals(CL, CR, center=True, want="frac")
    best = min(obs, key=lambda k: obs[k])
    best_fr = max(obs_fr, key=lambda k: obs_fr[k])

    # --- null A: shuffle block identities within the right hemisphere (sizes preserved)
    rng = np.random.default_rng(1000 * seed + hash(model) % 997)
    rb = R["block"].values
    rpq = R[["p", "q"]].values.astype(float)
    nullA_best, nullA_id, nullA_win, nullA_fr = [], [], [], []
    for _ in range(N_NULL):
        sb = rng.permutation(rb)
        Cs, _, _ = centroids(sb, rpq, MIN_BLOCK)
        res = all_transform_residuals(CL, Cs, center=True)
        frs = all_transform_residuals(CL, Cs, center=True, want="frac")
        nullA_best.append(min(res.values()))
        nullA_id.append(res["identity (p,q)"])
        nullA_win.append(res[best])
        nullA_fr.append(max(frs.values()))

    # --- null B: random one-to-one pairing of the *real* centroids (best transform applied)
    p, q = TRANSFORMS[best](CR[:, 0], CR[:, 1])
    CRb = np.c_[p, q]
    nullB = [random_pair_residual(CL, CRb, rng) for _ in range(N_NULL)]

    # --- unilateral-only sensitivity (drop blocks present in both hemispheres)
    keepL = ~np.isin(idL, idR)
    keepR = ~np.isin(idR, idL)
    obs_uni = all_transform_residuals(CL[keepL], CR[keepR], center=True) \
        if keepL.sum() and keepR.sum() else {k: np.nan for k in TRANSFORMS}

    return dict(model=model, seed=seed, n_blocks_total=nb_total,
                n_coord=len(df), nL=len(L), nR=len(R),
                kL=len(CL), kR=len(CR), bilateral=bilateral,
                best=best, best_fr=best_fr, obs=obs, obs_raw=obs_raw,
                obs_uni=obs_uni, obs_fr=obs_fr,
                nullA_fr_mean=float(np.mean(nullA_fr)), nullA_fr_sd=float(np.std(nullA_fr)),
                nullA_best_mean=float(np.mean(nullA_best)),
                nullA_best_sd=float(np.std(nullA_best)),
                nullA_win_mean=float(np.mean(nullA_win)),
                nullA_win_sd=float(np.std(nullA_win)),
                nullB_mean=float(np.mean(nullB)), nullB_sd=float(np.std(nullB)))


def main():
    col = pd.read_csv(COLFILE)
    labels = data.load_labels()
    side = labels["side"] if "side" in labels else None

    print("\n" + "=" * 88)
    print("COLUMN FILE: coordinate ranges per hemisphere (the trivial-explanation check)")
    print("=" * 88)
    print(col.groupby("hemisphere")[["p", "q"]].agg(["count", "min", "max", "mean"]).to_string())
    col["DV"] = col.p + col.q
    col["AP"] = col.p - col.q
    print(col.groupby("hemisphere")[["DV", "AP"]].agg(["min", "max", "mean"]).to_string())

    # cross-check hemisphere vs the project's own `side` label
    chk = col.join(side.rename("side"), on="root_id")
    print("\n  hemisphere (column file) x side (classification.csv):")
    print(pd.crosstab(chk["hemisphere"], chk["side"]).to_string())

    convention_audit(col)

    col = col[["root_id", "hemisphere", "p", "q"]]

    rows = []
    print("\n" + "=" * 88)
    print("PER-RUN MIRROR MATCHING  (residuals are mean Hungarian-matched centroid distance,")
    print("in hexagonal (p,q) units; right-hemisphere centroids re-centred on the left mean)")
    print("=" * 88)
    for model in MODELS:
        for seed in SEEDS:
            r = run_one(model, seed, col, side)
            if r is None:
                continue
            rows.append(r)
            print(f"\n{model:<12} s{seed}  blocks={r['n_blocks_total']:5d}  "
                  f"coords L/R={r['nL']}/{r['nR']}  centroids L/R={r['kL']}/{r['kR']}  "
                  f"bilateral={r['bilateral']}")
            for name in TRANSFORMS:
                mark = (" <== best-mean" if name == r["best"] else "") \
                    + (" <== best-frac" if name == r["best_fr"] else "")
                print(f"    {name:<18} centred={r['obs'][name]:6.3f}   "
                      f"raw={r['obs_raw'][name]:6.3f}   uni={r['obs_uni'][name]:6.3f}   "
                      f"frac<1={r['obs_fr'][name]:5.2f}{mark}")
            print(f"    NULL shuffle-blocks (best-of-transforms): {r['nullA_best_mean']:.3f} "
                  f"+/- {r['nullA_best_sd']:.3f}   |  same transform as obs: "
                  f"{r['nullA_win_mean']:.3f} +/- {r['nullA_win_sd']:.3f}   |  "
                  f"frac<1 (best): {r['nullA_fr_mean']:.3f} +/- {r['nullA_fr_sd']:.3f}")
            print(f"    NULL random-pairing of real centroids   : {r['nullB_mean']:.3f} "
                  f"+/- {r['nullB_sd']:.3f}")

    # -------------------------------------------------------------- summary
    print("\n" + "=" * 88)
    print("SUMMARY  (mean +/- sd over 5 seeds)")
    print("=" * 88)
    hdr = (f"{'model':<12}{'blocks':>8}{'k_L':>6}{'k_R':>6}  {'winner':<18}"
           f"{'obs':>8}{'null_shuf':>11}{'null_pair':>11}{'ratio':>8}"
           f"{'obs_f<1':>9}{'null_f<1':>10}")
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for model in MODELS:
        rs = [r for r in rows if r["model"] == model]
        if not rs:
            continue
        wins = pd.Series([r["best"] for r in rs]).value_counts()
        winner = wins.index[0]
        obs = np.array([r["obs"][winner] for r in rs])
        nsh = np.array([r["nullA_win_mean"] for r in rs])
        npr = np.array([r["nullB_mean"] for r in rs])
        ofr = np.array([r["obs_fr"][winner] for r in rs])
        nfr = np.array([r["nullA_fr_mean"] for r in rs])
        print(f"{model:<12}{np.mean([r['n_blocks_total'] for r in rs]):8.0f}"
              f"{np.mean([r['kL'] for r in rs]):6.0f}{np.mean([r['kR'] for r in rs]):6.0f}  "
              f"{winner:<18}{obs.mean():8.3f}{nsh.mean():11.3f}{npr.mean():11.3f}"
              f"{obs.mean()/nsh.mean():8.3f}{ofr.mean():9.3f}{nfr.mean():10.3f}")
        summary[model] = dict(winner=winner, wins=wins.to_dict(),
                              wins_frac=pd.Series([r["best_fr"] for r in rs]).value_counts().to_dict(),
                              obs=float(obs.mean()), obs_sd=float(obs.std()),
                              null_shuffle=float(nsh.mean()), null_pair=float(npr.mean()),
                              obs_frac=float(ofr.mean()), null_frac=float(nfr.mean()),
                              ratio=float(obs.mean() / nsh.mean()),
                              per_transform={t: float(np.mean([r["obs"][t] for r in rs]))
                                             for t in TRANSFORMS},
                              per_transform_frac={t: float(np.mean([r["obs_fr"][t] for r in rs]))
                                                  for t in TRANSFORMS})
    for key, lab in [("per_transform", "mean matched residual (lower = better)"),
                     ("per_transform_frac", "fraction of matched pairs within 1 hex unit (higher = better)")]:
        print(f"\nPer-transform, mean over seeds -- {lab}:")
        tw = max(len(t) for t in TRANSFORMS)
        print(f"{'transform':<{tw}}" + "".join(f"{m[:9]:>11}" for m in MODELS))
        for t in TRANSFORMS:
            print(f"{t:<{tw}}" + "".join(
                f"{summary[m][key][t]:11.3f}" if m in summary else f"{'-':>11}"
                for m in MODELS))

    out = Path("/var/tmp/csbm_work/results/t12_mirror_symmetry.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    sys.exit(main())
