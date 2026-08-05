"""
Does the weight model decide WHICH axis the SBM groups by -- cell-type identity or eye position?

For every block (optic lobe, one hemisphere at a time) we measure two things:

  spatial   = mean pairwise-to-centroid distance of its neurons        -> low  = tight in space
  identity  = number of distinct primary_types it contains             -> low  = type-pure

Both are trivially driven by block SIZE (a small block is tight and type-pure for free), and the
models differ ~8x in median block size, so raw values are meaningless. Each is therefore divided
by its expectation under a permutation that keeps the block-size distribution EXACTLY and
destroys only the assignment:

  spatial_ratio  = observed dispersion / permuted dispersion    (<1 => spatially organised)
  identity_ratio = observed n_types    / permuted n_types       (<1 => organised by cell type)

A model that groups by position: spatial_ratio << 1, identity_ratio ~ 1 (blocks mix types).
A model that groups by identity: identity_ratio << 1, spatial_ratio ~ 1 (blocks span the eye).

Gaussian (561 blocks) and geometric (563 blocks) sit at the same resolution, so the comparison
between those two is the control: any difference there is the weight distribution, not B.

Run from src/:  python spatial_vs_identity.py
"""
import argparse
import glob
import numpy as np
import pandas as pd
import config
import data

SCRATCH = str(config.SCRATCH_DIR)
MODELS = ["lognormal", "gaussian", "geometric", "poisson"]
N_PERM = 5
MIN_BLOCK = 5          # blocks smaller than this carry no usable dispersion estimate
RNG = np.random.default_rng(0)


def dispersion(coords, blk):
    """Mean distance-to-block-centroid, averaged over blocks (size-weighted).
    `coords` is (n, d): 3-D skeleton centroids, or 2-D published hex (p, q)."""
    cols = [f"c{i}" for i in range(coords.shape[1])]
    df = pd.DataFrame(coords, columns=cols)
    df["b"] = blk
    cen = df.groupby("b")[cols].transform("mean")
    df["d"] = np.linalg.norm(df[cols].values - cen.values, axis=1)
    sz = df.groupby("b").size()
    return df[df.b.isin(sz[sz >= MIN_BLOCK].index)].d.mean()


def n_types(types, blk):
    return pd.DataFrame({"t": types, "b": blk}).groupby("b").t.nunique().mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", choices=["centroid", "hex"], default="centroid",
                    help="centroid = 3-D skeleton centre of mass; "
                         "hex = published FlyWire column (p, q) from Matsliah et al. 2024")
    a = ap.parse_args()

    lab = data.load_labels()
    if a.coords == "hex":
        h = pd.read_csv(f"{SCRATCH}/column_assignment.csv.gz").set_index("root_id")
        h.index.name = "neuron"
        CO = ["p", "q"]
        base = lab.join(h[CO + ["column_id"]], how="inner")
    else:
        cen = pd.read_parquet(f"{SCRATCH}/centroids.parquet")
        CO = ["x", "y", "z"]
        base = lab.join(cen[CO], how="inner")
        base = base[base.super_class == "optic"]
    base = base[base.primary_type.notna() & base.side.isin(["left", "right"])]
    print(f"[{a.coords}] neurons with coordinate + type + side: {len(base):,}")

    rows = []
    for model in MODELS:
        for p in sorted(glob.glob(f"{config.WORK_DIR}/results/{model}_t5_dir_dc_s*/partition.npz")):
            z = np.load(p)
            s = pd.Series(z["blocks"], index=z["node_ids"], name="blk")
            df = base.join(s, how="inner").dropna(subset=["blk"])
            for side, g in df.groupby("side"):
                if len(g) < 500:
                    continue
                xyz = g[CO].values.astype(float)
                blk = g.blk.values
                typ = g.primary_type.values
                obs_s, obs_i = dispersion(xyz, blk), n_types(typ, blk)
                ps, pi = [], []
                for _ in range(N_PERM):
                    q = RNG.permutation(blk)          # keeps block sizes exactly
                    ps.append(dispersion(xyz, q))
                    pi.append(n_types(typ, q))
                rows.append(dict(model=model, seed=p.split("_s")[-1].split("/")[0], side=side,
                                 n=len(g), n_blocks=g.blk.nunique(),
                                 spatial_ratio=obs_s / np.mean(ps),
                                 identity_ratio=obs_i / np.mean(pi),
                                 med_block=g.groupby("blk").size().median()))
            print(f"  {model} {p.split('/')[-2]} done", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(f"{SCRATCH}/spatial_vs_identity_raw_{a.coords}.csv", index=False)
    summ = (r.groupby("model")
             .agg(blocks=("n_blocks", "mean"), med_block=("med_block", "mean"),
                  spatial_ratio=("spatial_ratio", "mean"), spatial_sd=("spatial_ratio", "std"),
                  identity_ratio=("identity_ratio", "mean"), identity_sd=("identity_ratio", "std"),
                  cells=("n", "size"))
             .sort_values("spatial_ratio"))
    print("\n" + "=" * 78)
    print("lower spatial_ratio  = blocks tighter in space than chance -> groups by POSITION")
    print("lower identity_ratio = blocks purer in type than chance    -> groups by IDENTITY")
    print("=" * 78)
    print(summ.round(3).to_string())
    print("\n--- matched-resolution control (gaussian 561 vs geometric 563 blocks) ---")
    print(summ.loc[[m for m in ["gaussian", "geometric"] if m in summ.index],
                   ["blocks", "spatial_ratio", "identity_ratio"]].round(3).to_string())
    print(f"\nraw -> {SCRATCH}/spatial_vs_identity_raw_{a.coords}.csv")


if __name__ == "__main__":
    main()
