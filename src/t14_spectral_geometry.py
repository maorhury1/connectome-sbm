"""
T14 -- does the geometry of the inferred BLOCK graph carry biological axes?

Everything so far treats the partition as a flat labelling. But the SBM also infers a
block-by-block connectivity matrix, and that matrix has a geometry of its own. If the weight
likelihood really shapes a biologically meaningful representation, the leading dimensions of
that block graph should line up with real biological axes rather than being arbitrary.

We spectrally embed the block graph (symmetrised, degree-normalised, leading non-trivial
eigenvectors of the normalised Laplacian) and ask whether the top dimensions align with:

  bilateral symmetry  block laterality = (n_left - n_right) / n
  retinotopy          block mean DV = p+q, and AP = p-q, from published hex coordinates
  cell identity       block's dominant super_class / dominant primary_type (categorical)

Scoring. For the two continuous axes we take the largest |Pearson r| over the leading K
dimensions; for identity we take the largest eta^2 (one-way ANOVA effect size) over the same K.
Taking a max over K dimensions inflates every statistic, so each is compared against a null in
which the BLOCK ATTRIBUTE is permuted across blocks and the same max is recomputed -- so the
null carries the identical selection bias. Reported as observed, null mean, and z.

Blocks are weighted by size nowhere: each block is one point in the embedding, which is the
object of interest. Blocks with fewer than MIN_BLOCK neurons, or without coordinates, are
dropped from the corresponding test only.

Run from src/:  python t14_spectral_geometry.py
"""
import glob
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh
import config
import data

MODELS = ["lognormal", "gaussian", "geometric", "exponential", "poisson"]
K_DIMS = 5          # leading non-trivial dimensions examined
N_PERM = 200
MIN_BLOCK = 10
RNG = np.random.default_rng(0)


def embed(wsum, k=K_DIMS):
    """Leading non-trivial eigenvectors of the normalised Laplacian of the block graph."""
    A = np.asarray(wsum, float)
    A = A + A.T                                   # direction is not the question here
    np.fill_diagonal(A, 0.0)
    d = A.sum(1)
    keep = d > 0
    A = A[np.ix_(keep, keep)]
    d = A.sum(1)
    Dm = sparse.diags(1.0 / np.sqrt(d))
    L = sparse.eye(A.shape[0]) - Dm @ sparse.csr_matrix(A) @ Dm
    kk = min(k + 1, A.shape[0] - 2)
    vals, vecs = eigsh(sparse.csr_matrix(L), k=kk, sigma=-1e-6, which="LM")
    order = np.argsort(vals)
    return vecs[:, order[1:]], keep          # drop the trivial constant eigenvector


def max_abs_corr(X, y):
    return max(abs(np.corrcoef(X[:, j], y)[0, 1]) for j in range(X.shape[1])
               if np.std(X[:, j]) > 0 and np.std(y) > 0)


def max_eta2(X, lab):
    """Largest one-way ANOVA eta^2 over dimensions, for a categorical block attribute."""
    best = 0.0
    codes = pd.Categorical(lab).codes
    for j in range(X.shape[1]):
        x = X[:, j]
        gm = x.mean()
        ss_t = ((x - gm) ** 2).sum()
        if ss_t <= 0:
            continue
        ss_b = sum(len(x[codes == c]) * (x[codes == c].mean() - gm) ** 2
                   for c in np.unique(codes) if (codes == c).sum() > 0)
        best = max(best, ss_b / ss_t)
    return best


def null_max(fn, X, y, n=N_PERM):
    v = [fn(X, RNG.permutation(y)) for _ in range(n)]
    return float(np.mean(v)), float(np.std(v))


def main():
    lab = data.load_labels()
    hexa = pd.read_csv(f"{config.SCRATCH_DIR}/column_assignment.csv.gz").set_index("root_id")
    hexa["DV"], hexa["AP"] = hexa.p + hexa.q, hexa.p - hexa.q

    rows = []
    for model in MODELS:
        for pdir in sorted(glob.glob(f"{config.WORK_DIR}/results/{model}_t5_dir_dc_s*")):
            try:
                bm = np.load(f"{pdir}/blockmat.npz")
                pz = np.load(f"{pdir}/partition.npz")
            except FileNotFoundError:
                continue
            blk = pd.Series(pz["blocks"], index=pz["node_ids"], name="blk")
            df = pd.DataFrame(blk).join(lab[["side", "super_class", "primary_type"]])
            df = df.join(hexa[["DV", "AP"]])

            bids = bm["block_ids"]
            X, keep = embed(bm["wsum"])
            bids = bids[keep]

            g = df.groupby("blk")
            size = g.size()
            lat = (g.side.apply(lambda s: (s == "left").sum() - (s == "right").sum()) / size)
            dv, ap = g.DV.mean(), g.AP.mean()
            sc = g.super_class.agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None)

            idx = pd.Index(bids)
            big = idx[size.reindex(idx).fillna(0).values >= MIN_BLOCK]
            sel = np.isin(bids, big)
            Xb = X[sel]
            b = bids[sel]

            def run(name, y_ser, fn):
                y = y_ser.reindex(b).values.astype(float) if fn is max_abs_corr \
                    else y_ser.reindex(b).values
                ok = pd.notna(y)
                if ok.sum() < 20:
                    return
                Xo, yo = Xb[ok], y[ok]
                obs = fn(Xo, yo)
                nm, ns = null_max(fn, Xo, yo)
                rows.append(dict(model=model, axis=name, n_blocks=int(ok.sum()),
                                 obs=obs, null=nm, z=(obs - nm) / ns if ns > 0 else np.nan))

            run("symmetry (laterality)", lat, max_abs_corr)
            run("retinotopy (DV)", dv, max_abs_corr)
            run("retinotopy (AP)", ap, max_abs_corr)
            run("identity (super_class)", sc, max_eta2)
            print(f"  {pdir.split('/')[-1]} done", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(f"{config.WORK_DIR}/results/t14_spectral.csv", index=False)
    print("\n" + "=" * 96)
    print(f"Alignment of the top {K_DIMS} block-graph dimensions with biological axes")
    print("obs = max |r| (or max eta^2) over dimensions; null = same statistic with the")
    print("attribute permuted across blocks, so the selection bias is matched. z = (obs-null)/sd")
    print("=" * 96)
    for axis, g in r.groupby("axis"):
        t = g.groupby("model").agg(n_blocks=("n_blocks", "mean"), obs=("obs", "mean"),
                                   null=("null", "mean"), z=("z", "mean"))
        t = t.reindex([m for m in MODELS if m in t.index]).sort_values("z", ascending=False)
        print(f"\n--- {axis} ---")
        print(t.round(3).to_string())


if __name__ == "__main__":
    main()
