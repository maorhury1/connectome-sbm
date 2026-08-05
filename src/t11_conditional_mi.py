"""
Do SBM blocks factorize identity x position, or do they only refine cell type?

The established result is that with a geometric weight model the blocks are spatially
localised and with a lognormal one they are not (unconditional AMI(block, column_id):
geometric 0.318, lognormal -0.076). That only shows "blocks are spatial", which is
already implied by "blocks are type-pure", because a cell type is itself a spatial
population. The factorization claim needs the stronger statement:

    knowing a neuron's block tells you where in the eye it sits
    EVEN AFTER you already know its cell type.

That is the conditional mutual information

    I(B ; C | T) = H(B,T) + H(C,T) - H(T) - H(B,C,T)

with B = block, C = retinotopic position, T = primary_type, computed per hemisphere.

TWO THINGS MAKE THE RAW NUMBER MEANINGLESS ON ITS OWN.

1. Plugin CMI is biased upward, badly, when the (B, C) table inside a type is sparse.
   Every number below is therefore reported against a permutation null that shuffles
   the block labels WITHIN each cell type. That keeps p(B|T) and p(C|T) exactly --
   same block-size distribution, same type composition, same position marginals --
   and destroys only the within-type block<->position association. excess = obs - null.

2. `column_id` cannot be used directly. The published column assignment is EXACTLY one
   neuron per (hemisphere, type, column) -- checked, 45,528/45,528. So within a cell
   type the column id is a unique neuron identifier, which forces

       I(B ; C | T) = H(B | T)      identically, for any block assignment whatsoever,

   and forces the within-type-shuffle null to the same value, so the excess is
   algebraically 0 for every model. That is a degeneracy of the variable, not a
   negative result. It is still computed and printed (POS=column_id) as the audit that
   this is what happens; the informative analysis coarsens position by binning the
   hexagonal (p, q) lattice into d x d super-columns, giving several resolutions with
   many neurons per (type, bin) cell.

Unconditional MI(B ; C) is reported alongside, against a global block shuffle, so the
cost of conditioning on identity is visible in the same units.

Run from src/:  python t11_conditional_mi.py
"""
import numpy as np
import pandas as pd
import config
import data

RESULTS = config.WORK_DIR / "results"
COLUMNS_CSV = config.SCRATCH_DIR / "column_assignment.csv.gz"

MODELS = ["lognormal", "gaussian", "geometric", "exponential", "poisson"]
SEEDS = [0, 1, 2, 3, 4]
TAG = "t5_dir_dc"

MIN_TYPE = 20          # a cell type needs this many neurons (with a column) to condition on
N_PERM = 50            # >= 20 required; 50 is cheap here and tightens the null s.d.
HEX_BINS = [1, 2, 4, 6, 8, 12]   # d = 1 is the raw column; d > 1 pools d x d hex columns
D_REF = 6              # reference resolution for the headline table / per-type breakdown
SEED0 = 20260805


# ---------------------------------------------------------------- entropy / MI


def _H(*codes):
    """Plugin entropy (nats) of the joint of several integer code arrays."""
    key = codes[0].astype(np.int64)
    for c in codes[1:]:
        key = key * (int(c.max()) + 1) + c.astype(np.int64)
    cnt = np.unique(key, return_counts=True)[1]
    p = cnt / cnt.sum()
    return float(-(p * np.log(p)).sum())


def cmi(b, c, t):
    """I(B ; C | T) in nats, plugin."""
    return _H(b, t) + _H(c, t) - _H(t) - _H(b, c, t)


def mi(b, c):
    """I(B ; C) in nats, plugin."""
    return _H(b) + _H(c) - _H(b, c)


def shuffle_within(b, t, rng):
    """Permute b inside each level of t. Preserves p(B|T) exactly."""
    order = np.lexsort((rng.random(b.size), t))     # group by t, random inside group
    out = np.empty_like(b)
    out[np.argsort(t, kind="stable")] = b[order]
    return out


# ---------------------------------------------------------------- data


def load_frame():
    """One row per neuron that has a block, a column and a primary_type."""
    col = pd.read_csv(COLUMNS_CSV)
    lab = data.load_labels()
    df = col.merge(lab[["primary_type"]], left_on="root_id", right_index=True, how="left")
    df = df.dropna(subset=["primary_type"])
    # position variables: raw column, plus d x d hex super-columns (per hemisphere)
    for d in HEX_BINS:
        df[f"pos{d}"] = (df.p // d).astype(str) + "_" + (df.q // d).astype(str)
    return df


def load_partition(model, seed):
    z = np.load(RESULTS / f"{model}_{TAG}_s{seed}" / "partition.npz")
    return pd.Series(z["blocks"], index=z["node_ids"])


# ---------------------------------------------------------------- main


def main():
    df = load_frame()
    print(f"[t11] {len(df):,} neurons with column + primary_type")
    g = df.groupby(["hemisphere", "primary_type", "column_id"]).size()
    print(f"[t11] neurons per (hemisphere, type, column): mean {g.mean():.3f} max {g.max()} "
          f"-> column_id is a within-type unique id, CMI on it is degenerate (see docstring)")

    rows, diag = [], []
    for model in MODELS:
        for seed in SEEDS:
            blocks = load_partition(model, seed)
            for hemi, h in df.groupby("hemisphere"):
                h = h[h.root_id.isin(blocks.index)]
                b_all = blocks.loc[h.root_id].to_numpy()
                # types with enough members, inside this hemisphere
                keep = h.primary_type.map(h.primary_type.value_counts()) >= MIN_TYPE
                b = b_all[keep.to_numpy()]
                hk = h[keep.to_numpy()]
                t = pd.factorize(hk.primary_type)[0]
                bb = pd.factorize(b)[0]
                n_types = int(hk.primary_type.nunique())

                for d in HEX_BINS:
                    cc = pd.factorize(hk[f"pos{d}"])[0]
                    obs_c, obs_m = cmi(bb, cc, t), mi(bb, cc)
                    rng = np.random.default_rng(SEED0 + 7 * seed + d)
                    nc, nm = [], []
                    for _ in range(N_PERM):
                        p_in = shuffle_within(bb, t, rng)
                        nc.append(cmi(p_in, cc, t))
                        nm.append(mi(rng.permutation(bb), cc))
                    rows.append(dict(
                        model=model, seed=seed, hemi=hemi, d=d, n=len(bb),
                        n_types=n_types, n_pos=int(cc.max() + 1), n_blocks=int(bb.max() + 1),
                        H_pos_given_type=_H(cc, t) - _H(t), H_pos=_H(cc),
                        H_blk_given_type=_H(bb, t) - _H(t),
                        cmi_obs=obs_c, cmi_null=float(np.mean(nc)), cmi_null_sd=float(np.std(nc)),
                        mi_obs=obs_m, mi_null=float(np.mean(nm)), mi_null_sd=float(np.std(nm)),
                    ))
                if seed == 0:
                    # per-type breakdown at the reference resolution: is the excess carried by
                    # all cell types, or by a handful? Same within-type-shuffle null, per type.
                    cc = pd.factorize(hk[f"pos{D_REF}"])[0]
                    rng = np.random.default_rng(SEED0 + 99)
                    perms = [shuffle_within(bb, t, rng) for _ in range(N_PERM)]
                    pos_t = 0
                    for ti in range(n_types):
                        m = t == ti
                        o = mi(bb[m], cc[m])
                        nul = np.mean([mi(p[m], cc[m]) for p in perms])
                        pos_t += (o - nul) > 0.05
                    diag.append(dict(model=model, hemi=hemi, n_types=n_types,
                                     blocks_per_type=float(pd.Series(bb).groupby(t).nunique().mean()),
                                     neurons_per_type=len(bb) / max(n_types, 1),
                                     types_with_excess=pos_t))

    r = pd.DataFrame(rows)
    r["cmi_excess"] = r.cmi_obs - r.cmi_null
    r["mi_excess"] = r.mi_obs - r.mi_null
    r["cmi_excess_frac"] = r.cmi_excess / r.H_pos_given_type
    r["mi_excess_frac"] = r.mi_excess / r.H_pos
    r["cmi_z"] = r.cmi_excess / r.cmi_null_sd.replace(0, np.nan)
    out = RESULTS / "t11_conditional_mi.csv"
    r.to_csv(out, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("\n=== diagnostics (seed 0) ===")
    print(pd.DataFrame(diag).to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    for d in HEX_BINS:
        sub = r[r.d == d]
        lab = "column_id (raw, DEGENERATE)" if d == 1 else f"{d}x{d} hex super-columns"
        print(f"\n=== POS = {lab}   K~{sub.n_pos.mean():.0f} bins, "
              f"{sub.n.mean() / sub.n_types.mean() / sub.n_pos.mean():.1f} neurons per (type,bin) ===")
        for hemi in ["left", "right"]:
            s = sub[sub.hemi == hemi].groupby("model").mean(numeric_only=True).reindex(MODELS)
            s = s[["n_blocks", "H_pos_given_type", "cmi_obs", "cmi_null", "cmi_excess",
                   "cmi_excess_frac", "cmi_z", "mi_obs", "mi_null", "mi_excess", "mi_excess_frac"]]
            print(f"-- {hemi} hemisphere (mean of {len(SEEDS)} seeds) --")
            print(s.to_string(float_format=lambda v: f"{v:8.3f}"))
    ref = r[r.d == D_REF]
    print(f"\n=== HEADLINE: d={D_REF} ({ref.n_pos.mean():.0f} bins), pooled over "
          f"{len(SEEDS)} seeds x 2 hemispheres, mean +- sd ===")
    print(f"H(pos)={ref.H_pos.mean():.3f}  H(pos|type)={ref.H_pos_given_type.mean():.3f}  "
          f"=> I(type;pos)={ref.H_pos.mean() - ref.H_pos_given_type.mean():.4f} nats "
          f"(cell type says almost nothing about column: every columnar type tiles the whole eye)")
    agg = ref.groupby("model").agg(
        cmi_obs=("cmi_obs", "mean"), cmi_null=("cmi_null", "mean"),
        cmi_excess=("cmi_excess", "mean"), cmi_excess_sd=("cmi_excess", "std"),
        frac_of_Hposgivtype=("cmi_excess_frac", "mean"), z=("cmi_z", "mean"),
        mi_obs=("mi_obs", "mean"), mi_null=("mi_null", "mean"),
        mi_excess=("mi_excess", "mean"),
    ).reindex(MODELS)
    agg["cmi_over_mi"] = agg.cmi_obs / agg.mi_obs
    print(agg.to_string(float_format=lambda v: f"{v:8.3f}"))
    print(f"\n[t11] wrote {out}")


if __name__ == "__main__":
    main()
