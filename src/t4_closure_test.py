"""
T4 -- do Piazza et al.'s closure identities hold WITHIN homogeneous groups?

Their Eq. (1) says a neuron's synapse count is set by its physical size, S ~ rho * L, with L the
total cable length and rho the LOCAL synapse density. If log L and log rho are jointly Gaussian
then S is lognormal with

    mu_S    = mu_L + mu_rho                                     (Eq. 3)
    sigma_S^2 = sigma_L^2 + sigma_rho^2 + 2 r sigma_L sigma_rho (Eq. 4)

They verify these once per connectome, i.e. pooled over every neuron. Pooled agreement is weak
evidence: Eq. (3) is close to an identity whenever S ~ rho*L holds at all, and pooling mixes
cell types with very different sizes. The sharper question is whether the identities survive
WITHIN a cell type and within an SBM block.

CRITICAL: rho must be measured INDEPENDENTLY of S. Defining rho := S / L makes
log S = log L + log rho true by construction and both identities become vacuous. So rho is
measured from synapse POSITIONS, following their SI 3.2: for each synapse of a neuron, count how
many of that neuron's synapses lie within 10 um, and average over the neuron's synapses.

Run from src/:  python t4_closure_test.py
"""
import glob
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import config
import data

SCRATCH = str(config.SCRATCH_DIR)
SYN = (config.DATA_DIR / "fafb_v783_princeton_synapse_table.csv.gz")
ID_PREFIX = 720575940 * 10**9          # the table stores root ids with this prefix stripped
RADIUS_NM = 10_000.0                   # 10 um, as in their SI 3.2
MIN_SYN = 20                           # neurons with too few synapses give an unstable rho
MIN_GROUP = 30
SAMPLE_PER_TYPE = 120                  # cap per cell type: keeps the synapse scan tractable
RNG = np.random.default_rng(0)


def local_density(xyz):
    """Mean number of the neuron's own synapses within RADIUS_NM of each of its synapses."""
    t = cKDTree(xyz)
    return float(np.mean([len(v) for v in t.query_ball_point(xyz, RADIUS_NM)]))


def build_rho(target_ids):
    """Stream the synapse table, keeping only synapses belonging to `target_ids`."""
    keep = np.array(sorted(target_ids), dtype=np.int64)
    kset = set(keep.tolist())
    chunks = []
    cols = ["ctr_x", "ctr_y", "ctr_z", "pre_root_id_720575940", "post_root_id_720575940"]
    n_read = 0
    for ch in pd.read_csv(SYN, usecols=cols, chunksize=4_000_000):
        n_read += len(ch)
        pre = ch["pre_root_id_720575940"].astype("int64") + ID_PREFIX
        post = ch["post_root_id_720575940"].astype("int64") + ID_PREFIX
        xyz = ch[["ctr_x", "ctr_y", "ctr_z"]].to_numpy(np.float32)
        for ids in (pre, post):
            m = ids.isin(kset).to_numpy()
            if m.any():
                chunks.append(pd.DataFrame({"id": ids.to_numpy()[m],
                                            "x": xyz[m, 0], "y": xyz[m, 1], "z": xyz[m, 2]}))
        print(f"    scanned {n_read:,} synapses, kept {sum(len(c) for c in chunks):,}", flush=True)
    syn = pd.concat(chunks, ignore_index=True)
    rows = []
    for nid, g in syn.groupby("id"):
        if len(g) >= MIN_SYN:
            rows.append((nid, local_density(g[["x", "y", "z"]].to_numpy(np.float64)), len(g)))
    return pd.DataFrame(rows, columns=["neuron", "rho", "n_syn"]).set_index("neuron")


def closure(d):
    """Observed vs predicted (mu_S, sigma_S) from Eqs. (3)-(4) for one group."""
    lS, lL, lr = np.log(d.S.values), np.log(d.L.values), np.log(d.rho.values)
    if len(d) < MIN_GROUP or min(lS.std(), lL.std(), lr.std()) <= 0:
        return None
    r = np.corrcoef(lL, lr)[0, 1]
    mu_pred = lL.mean() + lr.mean()
    sd_pred = np.sqrt(max(lL.var() + lr.var() + 2 * r * lL.std() * lr.std(), 1e-12))
    return dict(n=len(d), mu_obs=lS.mean(), mu_pred=mu_pred, mu_err=mu_pred - lS.mean(),
                sd_obs=lS.std(), sd_pred=sd_pred, sd_ratio=sd_pred / lS.std(),
                r_Lrho=r, corr_S_rhoL=np.corrcoef(lS, lL + lr)[0, 1])


def main():
    lab = data.load_labels()
    c = pd.read_parquet(f"{SCRATCH}/conn.parquet")
    ends = pd.concat([c.rename(columns={"pre_root_id": "id"})[["id", "syn_count"]],
                      c.rename(columns={"post_root_id": "id"})[["id", "syn_count"]]])
    S = ends.groupby("id").syn_count.sum().astype(float).rename("S")

    morph = pd.read_feather(config.DATA_DIR.parent.parent / "tree_morphology"
                            / "simplyfied_neural_structures.ftr",
                            columns=["neuron", "cable_length"])
    L = morph.groupby("neuron").cable_length.first().rename("L")

    df = pd.concat([S, L, lab.primary_type, lab.side], axis=1).dropna(subset=["S", "L"])
    df = df[(df.S > 0) & (df.L > 0)]
    print(f"neurons with S and L: {len(df):,}")

    # sample, capped per type, so the synapse scan stays tractable
    have = df[df.primary_type.notna()]
    idx = np.concatenate([g.sample(min(len(g), SAMPLE_PER_TYPE), random_state=0).index.values
                          for _, g in have.groupby("primary_type")])
    samp = df.loc[idx]
    print(f"sampled {len(samp):,} neurons across {samp.primary_type.nunique()} types; scanning synapses...")

    rho_path = f"{SCRATCH}/rho.parquet"
    try:
        rho = pd.read_parquet(rho_path)
        print(f"  reusing cached rho for {len(rho):,} neurons")
    except Exception:
        rho = build_rho(set(samp.index.tolist()))
        rho.to_parquet(rho_path)
    d = samp.join(rho, how="inner")
    print(f"neurons with S, L and independently measured rho: {len(d):,}")

    z = np.load(f"{config.WORK_DIR}/results/lognormal_t5_dir_dc_s0/partition.npz")
    d["blk"] = pd.Series(z["blocks"], index=z["node_ids"]).reindex(d.index)

    print(f"\nsanity: corr(log S, log rho + log L) pooled = "
          f"{np.corrcoef(np.log(d.S), np.log(d.rho) + np.log(d.L))[0,1]:.3f}   "
          f"(their Eq. 1; rho measured independently, so this is NOT automatic)")

    out = [dict(level="ALL (pooled)", **closure(d))]
    for lv, key in [("by TYPE", "primary_type"), ("by BLOCK", "blk")]:
        rows = [closure(g) for _, g in d.dropna(subset=[key]).groupby(key)]
        rows = [r for r in rows if r]
        if rows:
            g = pd.DataFrame(rows)
            w = g.n / g.n.sum()
            out.append(dict(level=lv, n=int(g.n.sum()), n_groups=len(g),
                            mu_err=float((g.mu_err * w).sum()),
                            sd_ratio=float((g.sd_ratio * w).sum()),
                            r_Lrho=float((g.r_Lrho * w).sum()),
                            corr_S_rhoL=float((g.corr_S_rhoL * w).sum())))
    r = pd.DataFrame(out)
    r.to_csv(f"{SCRATCH}/t4_closure.csv", index=False)
    print("\n" + "=" * 92)
    print("mu_err   = predicted mu_S - observed mu_S    (Eq. 3; 0 = holds)")
    print("sd_ratio = predicted sigma_S / observed      (Eq. 4; 1 = holds)")
    print("=" * 92)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
