"""
T8 -- label-free bilateral symmetry: do blocks merge the two hemispheres, or split them?

The only label used is `side` (left / right). Cell types are NOT used, so this cannot inherit
the circularity of a connectivity-derived annotation.

The logic. The two hemispheres are joined by very few edges (measured below). An SBM groups by
connectivity, so the path of least resistance is to SPLIT every group in two -- one block for
the left copy, one for the right -- because it never has to explain any edges between them.
Merging a left group with its right counterpart is only worthwhile if the model recognises that
the two have the SAME connection pattern despite sharing almost no edges. So:

    blocks are one-sided  -> the model split the brain down the midline (the easy answer)
    blocks are two-sided  -> the model recognised bilateral homology (the informative answer)

Two measures:

  AMI(block, side)   how much a block tells you about which hemisphere a neuron is in.
                     LOW = symmetric (block is uninformative about side)
                     HIGH = hemispheres were separated
  % bilateral        share of neurons in blocks where the minority side is >= 20%

AMI is near 0 for a RANDOM partition too, so on its own it is not evidence. It becomes evidence
in combination with the already-established fact that these same partitions are highly
informative about cell type: a partition that predicts type well but side poorly is bilaterally
symmetric, not random. Both numbers are therefore reported together.

Run from src/:  python t8_symmetry_labelfree.py
"""
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score as ami
import config
import data

MODELS = ["lognormal", "gaussian", "geometric", "exponential", "poisson"]
MINORITY = 0.20


def main():
    lab = data.load_labels()
    side = lab.side[lab.side.isin(["left", "right"])]

    # how separable are the hemispheres to begin with?
    pre, post, w = data.load_edges(threshold=5, directed=True)
    s_pre = side.reindex(pre).values
    s_post = side.reindex(post).values
    both = pd.notna(s_pre) & pd.notna(s_post)
    contra = (s_pre[both] != s_post[both])
    print(f"edges with both endpoints sided: {both.sum():,}")
    print(f"  contralateral (cross-midline): {contra.sum():,} = {100*contra.mean():.2f}%")
    print("  -> splitting the brain by hemisphere leaves almost no edges unexplained,")
    print("     so a one-sided partition is the easy answer and a two-sided one is not.\n")

    rows = []
    for m in MODELS:
        for p in sorted(glob.glob(f"{config.WORK_DIR}/results/{m}_t5_dir_dc_s*/partition.npz")):
            z = np.load(p)
            b = pd.Series(z["blocks"], index=z["node_ids"])
            d = pd.DataFrame({"blk": b}).join(side.rename("side"), how="inner").dropna()
            ct = pd.crosstab(d.blk, d.side)
            if not {"left", "right"} <= set(ct.columns):
                continue
            tot = ct.sum(axis=1)
            minority = ct.min(axis=1) / tot
            bilateral = minority >= MINORITY
            rows.append(dict(model=m, n=len(d), n_blocks=len(ct),
                             ami_side=ami(d.side.astype(str), d.blk.astype(int)),
                             pct_neurons_bilateral=100 * tot[bilateral].sum() / tot.sum(),
                             pct_blocks_bilateral=100 * bilateral.mean(),
                             mean_minority_share=float((minority * tot).sum() / tot.sum())))
    r = pd.DataFrame(rows)
    summ = r.groupby("model").agg(blocks=("n_blocks", "mean"),
                                  AMI_side=("ami_side", "mean"), AMI_sd=("ami_side", "std"),
                                  pct_neurons_bilateral=("pct_neurons_bilateral", "mean"),
                                  pct_blocks_bilateral=("pct_blocks_bilateral", "mean"),
                                  minority_share=("mean_minority_share", "mean"))
    summ = summ.reindex([m for m in MODELS if m in summ.index]).sort_values("AMI_side")
    print("=" * 100)
    print("AMI(block, side): LOW = blocks merge the hemispheres (symmetric)")
    print("                  HIGH = blocks split them (midline partition)")
    print("=" * 100)
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
