"""
T13 -- how well does each weight model recover a cell type, as a function of TYPE SIZE?

Reviewer objection this answers: "a ~190-block partition cannot recover 8,600 cell types,
so the AMI result is meaningless". The honest claim is that very small types have weak
statistical support under an automatically regularised global model. That has to be SHOWN
as a curve over type size, not asserted.

Score: per-type best-match Dice/F1. For type T, over all blocks B,
    Dice(T) = max_B 2|T n B| / (|T| + |B|)
plus purity = |T n B*| / |B*| and recall = |T n B*| / |T| for the argmax block B*.

CRITICAL CONTROL: max-over-blocks Dice is biased upward when a partition has more blocks
(more chances to match), and these models differ ~8x in block count. So the same statistic is
recomputed on N_PERM label shuffles that permute the block assignment across neurons
(block sizes exactly preserved). Everything is reported observed, null, observed-null, and
observed/null.

Universe: neurons that are in the partition AND have a primary_type AND side in {left,right}.
Types and blocks are both restricted to one hemisphere at a time, so a bilateral block is not
punished for containing the contralateral copy of the same type. Type sizes are therefore
per-hemisphere counts. Size-1 types are excluded (the pre-declared bins start at 2).

Bins were declared BEFORE looking at any result and are not to be tuned: 2-3, 4-7, 8-15,
16-31, 32+.

Run from src/:  python t13_recovery_by_size.py
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

import data

RESULTS = "/var/tmp/csbm_work/results"
MODELS = ["lognormal", "gaussian", "geometric", "exponential", "poisson"]
SEEDS = [0, 1, 2, 3, 4]
SIDES = ["left", "right"]
N_PERM = 20
BIN_EDGES = [2, 4, 8, 16, 32, np.inf]          # 2-3, 4-7, 8-15, 16-31, 32+
BIN_NAMES = ["2-3", "4-7", "8-15", "16-31", "32+"]
OUT_CSV = "/var/tmp/csbm_work/results/t13_recovery_by_size.csv"


def best_match(type_idx, block_idx, type_sizes, block_sizes):
    """Per-type best-matching-block Dice, purity, recall.

    type_idx/block_idx are aligned per-neuron index arrays over one hemisphere universe.
    Only blocks that actually intersect a type can win (Dice is 0 otherwise), so we walk the
    sparse contingency table row by row.
    """
    n_t, n_b = len(type_sizes), len(block_sizes)
    C = sp.coo_matrix((np.ones(len(type_idx), np.int32), (type_idx, block_idx)),
                      shape=(n_t, n_b)).tocsr()
    dice = np.zeros(n_t)
    pur = np.zeros(n_t)
    rec = np.zeros(n_t)
    indptr, indices, dat = C.indptr, C.indices, C.data
    for t in range(n_t):
        s, e = indptr[t], indptr[t + 1]
        if s == e:
            continue
        cnt = dat[s:e].astype(np.float64)
        bl = indices[s:e]
        d = 2.0 * cnt / (type_sizes[t] + block_sizes[bl])
        j = int(np.argmax(d))
        dice[t] = d[j]
        pur[t] = cnt[j] / block_sizes[bl[j]]
        rec[t] = cnt[j] / type_sizes[t]
    return dice, pur, rec


def main():
    lab = data.load_labels()
    rows = []
    per_type_rows = []

    for model in MODELS:
        for seed in SEEDS:
            p = np.load(f"{RESULTS}/{model}_t5_dir_dc_s{seed}/partition.npz")
            nid, blocks = p["node_ids"], p["blocks"]
            sub = lab.reindex(nid)
            n_blocks_total = len(np.unique(blocks))

            for side in SIDES:
                keep = (sub["side"].values == side) & sub["primary_type"].notna().values
                b_raw = blocks[keep]
                t_raw = sub["primary_type"].values[keep]

                t_codes, t_names = pd.factorize(t_raw)
                b_codes, _ = pd.factorize(b_raw)
                type_sizes = np.bincount(t_codes).astype(np.float64)
                block_sizes = np.bincount(b_codes).astype(np.float64)

                dice, pur, rec = best_match(t_codes, b_codes, type_sizes, block_sizes)

                # null: permute block labels over neurons, block sizes preserved exactly
                rng = np.random.default_rng(1000 * (MODELS.index(model) + 1) + 10 * seed +
                                            SIDES.index(side))
                nd = np.zeros_like(dice)
                npu = np.zeros_like(pur)
                nre = np.zeros_like(rec)
                for _ in range(N_PERM):
                    perm = rng.permutation(b_codes)
                    d0, p0, r0 = best_match(t_codes, perm, type_sizes, block_sizes)
                    nd += d0
                    npu += p0
                    nre += r0
                nd /= N_PERM
                npu /= N_PERM
                nre /= N_PERM

                sel = type_sizes >= 2
                bin_idx = np.digitize(type_sizes, BIN_EDGES) - 1
                for bi, bname in enumerate(BIN_NAMES):
                    m = sel & (bin_idx == bi)
                    if not m.any():
                        continue
                    rows.append(dict(
                        model=model, seed=seed, side=side, size_bin=bname,
                        n_types=int(m.sum()), n_blocks=n_blocks_total,
                        dice=dice[m].mean(), purity=pur[m].mean(), recall=rec[m].mean(),
                        null_dice=nd[m].mean(), null_purity=npu[m].mean(),
                        null_recall=nre[m].mean()))
                if seed == 0:
                    for t in np.where(sel)[0]:
                        per_type_rows.append(dict(
                            model=model, side=side, primary_type=t_names[t],
                            size=int(type_sizes[t]), dice=dice[t], purity=pur[t],
                            recall=rec[t], null_dice=nd[t]))
            print(f"[t13] {model} s{seed}: {n_blocks_total} blocks -- done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    pt = pd.DataFrame(per_type_rows)
    pt.to_csv(OUT_CSV.replace(".csv", "_pertype_s0.csv"), index=False)

    # ---- aggregate over seeds and hemispheres --------------------------------------------
    w = df.groupby(["model", "size_bin"]).agg(
        n_types=("n_types", "mean"), n_blocks=("n_blocks", "mean"),
        dice=("dice", "mean"), purity=("purity", "mean"), recall=("recall", "mean"),
        null_dice=("null_dice", "mean"), null_purity=("null_purity", "mean"),
        null_recall=("null_recall", "mean"),
        dice_sd=("dice", "std")).reset_index()
    w["excess"] = w["dice"] - w["null_dice"]
    w["ratio"] = w["dice"] / w["null_dice"]

    order = {b: i for i, b in enumerate(BIN_NAMES)}
    w = w.sort_values(["model", "size_bin"], key=lambda s: s.map(order) if s.name == "size_bin" else s)

    pd.set_option("display.width", 200)
    print("\n=== per-type best-match recovery, mean over 5 seeds x 2 hemispheres ===")
    print(f"(N_PERM={N_PERM} block-label shuffles per seed/side; n_types is per hemisphere)")
    for model in MODELS:
        sub = w[w.model == model]
        print(f"\n{model}  (mean blocks = {sub.n_blocks.iloc[0]:.0f})")
        print(f"{'bin':>7} {'nTypes':>7} {'Dice':>6} {'Pur':>6} {'Rec':>6} | "
              f"{'nullD':>6} {'nullP':>6} {'nullR':>6} | {'D-null':>7} {'D/null':>7}")
        for _, r in sub.iterrows():
            print(f"{r.size_bin:>7} {r.n_types:7.0f} {r.dice:6.3f} {r.purity:6.3f} "
                  f"{r.recall:6.3f} | {r.null_dice:6.3f} {r.null_purity:6.3f} "
                  f"{r.null_recall:6.3f} | {r.excess:7.3f} {r.ratio:7.2f}")

    print("\n=== observed Dice by bin (models as columns) ===")
    print(w.pivot(index="size_bin", columns="model", values="dice").reindex(BIN_NAMES).round(3))
    print("\n=== null-corrected excess Dice (obs - null) ===")
    print(w.pivot(index="size_bin", columns="model", values="excess").reindex(BIN_NAMES).round(3))
    print("\n=== ratio obs/null ===")
    print(w.pivot(index="size_bin", columns="model", values="ratio").reindex(BIN_NAMES).round(2))

    # spread across models per bin, and seed-level sd, to judge "do models differ"
    print("\n=== per-bin: max-min excess across models vs typical seed/side sd of Dice ===")
    for b in BIN_NAMES:
        sb = w[w.size_bin == b]
        sd = df[df.size_bin == b].groupby("model")["dice"].std().mean()
        print(f"{b:>7}: excess range {sb.excess.min():.3f}..{sb.excess.max():.3f} "
              f"(spread {sb.excess.max() - sb.excess.min():.3f}), "
              f"best={sb.loc[sb.excess.idxmax(), 'model']}, "
              f"within-model sd over seeds/sides = {sd:.3f}")

    # how many types are 'well recovered' (Dice >= 0.5) per bin
    print("\n=== fraction of types with observed Dice >= 0.5 (seed 0, both hemispheres) ===")
    pt["size_bin"] = pd.cut(pt["size"], bins=[2, 4, 8, 16, 32, np.inf], right=False,
                            labels=BIN_NAMES)
    tab = pt.groupby(["model", "size_bin"], observed=True).apply(
        lambda g: (g.dice >= 0.5).mean(), include_groups=False).unstack()
    print(tab.reindex(columns=BIN_NAMES).round(3))
    print(f"\n[t13] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
