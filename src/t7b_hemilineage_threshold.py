"""
T7b -- is the hemilineage tie an artefact of tiny label classes?

In T7 every weight model scored ~0.48-0.51 AMI against hemilineage, a four-way tie that
gaussian "won" by 0.01. But no minimum class size was applied, so lineages represented by a
handful of neurons were included and may be washing out real differences.

This sweeps a minimum members-per-class threshold. Classes below it are dropped (not merged),
and the AMI is recomputed on the surviving neurons only.

Run from src/:  python t7b_hemilineage_threshold.py
"""
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score
import config
import data

SCRATCH = str(config.SCRATCH_DIR)
MODELS = ["lognormal", "gaussian", "geometric", "exponential", "poisson"]
THRESHOLDS = [1, 5, 10, 20, 50, 100]


def main():
    lab = data.load_labels()
    ann = pd.read_csv(f"{SCRATCH}/flywire_ann.tsv", sep="\t", low_memory=False,
                      usecols=["root_id", "ito_lee_hemilineage",
                               "hartenstein_hemilineage"]).set_index("root_id")
    labelings = {"ito-lee": ann.ito_lee_hemilineage,
                 "hartenstein": ann.hartenstein_hemilineage}
    side = lab.side

    parts = {}
    for m in MODELS:
        for p in sorted(glob.glob(f"{config.WORK_DIR}/results/{m}_t5_dir_dc_s*/partition.npz")):
            z = np.load(p)
            parts.setdefault(m, []).append(pd.Series(z["blocks"], index=z["node_ids"]))

    rows = []
    for lname, lser in labelings.items():
        for hemi in ["left", "right"]:
            ids = side.index[side == hemi]
            y_full = lser.reindex(ids).dropna()
            for thr in THRESHOLDS:
                vc = y_full.value_counts()
                keep = vc[vc >= thr].index
                y = y_full[y_full.isin(keep)]
                if y.nunique() < 2 or len(y) < 200:
                    continue
                for m, ps in parts.items():
                    for blk in ps:
                        b = blk.reindex(y.index).dropna()
                        yy = y.reindex(b.index)
                        rows.append(dict(labeling=lname, side=hemi, min_class=thr, model=m,
                                         n=len(yy), n_classes=yy.nunique(),
                                         AMI=adjusted_mutual_info_score(yy.astype(str),
                                                                        b.astype(int))))
    r = pd.DataFrame(rows)
    r.to_csv(f"{SCRATCH}/t7b_hemilineage.csv", index=False)

    print("=" * 104)
    print("AMI vs hemilineage, sweeping the minimum members-per-class threshold")
    print("=" * 104)
    for lname in labelings:
        d = r[r.labeling == lname]
        piv = d.pivot_table(index="min_class", columns="model", values="AMI", aggfunc="mean")
        piv = piv[[m for m in MODELS if m in piv.columns]]
        meta = d.groupby("min_class").agg(n=("n", "mean"), classes=("n_classes", "mean")).round(0)
        piv["WINNER"] = piv.idxmax(axis=1)
        piv["margin"] = (piv[MODELS].max(axis=1) - piv[MODELS].apply(
            lambda s: s.nlargest(2).iloc[-1], axis=1))
        print(f"\n--- hemilineage: {lname} ---")
        print(piv.join(meta).round(3).to_string())


if __name__ == "__main__":
    main()
