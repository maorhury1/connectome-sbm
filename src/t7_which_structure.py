"""
T7 -- which biological structure does each weight model recover?

We have two results already: with a lognormal weight model the SBM's blocks line up with CELL
TYPE, with a geometric one they line up with the RETINOTOPIC MAP. The obvious question is
whether other biological labelings are recovered by other weight models, or whether everything
else is just cell type in disguise.

Labelings tested (all evaluation-only, never seen during fitting):

  cell type       primary_type                     -- wiring identity   (known: lognormal)
  retinotopic map published hex column_id          -- eye position      (known: geometric)
  neurotransmitter top_nt                          -- chemical identity
  hemilineage     ito_lee / hartenstein            -- developmental origin
  neuropil        dominant neuropil by synapse count -- anatomical compartment

Scored by adjusted mutual information, which corrects for chance and is not inflated by a model
simply producing more blocks -- essential here, since the models differ ~8x in block count.
Reported per hemisphere and averaged over seeds.

Run from src/:  python t7_which_structure.py
"""
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, v_measure_score
import config
import data

SCRATCH = str(config.SCRATCH_DIR)
MODELS = ["lognormal", "gaussian", "geometric", "exponential", "poisson"]
MIN_LABELLED = 300


def load_labelings():
    lab = data.load_labels()
    out = {"cell type": lab.primary_type}

    ann = pd.read_csv(f"{SCRATCH}/flywire_ann.tsv", sep="\t", low_memory=False,
                      usecols=["root_id", "top_nt", "ito_lee_hemilineage",
                               "hartenstein_hemilineage"]).set_index("root_id")
    out["neurotransmitter"] = ann.top_nt
    out["hemilineage (ito-lee)"] = ann.ito_lee_hemilineage
    out["hemilineage (hartenstein)"] = ann.hartenstein_hemilineage

    hexa = pd.read_csv(f"{SCRATCH}/column_assignment.csv.gz").set_index("root_id")
    out["retinotopic column"] = hexa.column_id

    # dominant neuropil: the compartment holding most of a neuron's synapses
    c = pd.read_csv(config.CONNECTIONS, usecols=["pre_root_id", "post_root_id",
                                                 "neuropil", "syn_count"])
    e = pd.concat([c.rename(columns={"pre_root_id": "id"})[["id", "neuropil", "syn_count"]],
                   c.rename(columns={"post_root_id": "id"})[["id", "neuropil", "syn_count"]]])
    e = e.dropna(subset=["neuropil"])
    g = e.groupby(["id", "neuropil"]).syn_count.sum().reset_index()
    out["neuropil"] = g.loc[g.groupby("id").syn_count.idxmax()].set_index("id").neuropil

    return lab, out


def main():
    lab, labelings = load_labelings()
    side = lab.side
    rows = []
    for model in MODELS:
        for p in sorted(glob.glob(f"{config.WORK_DIR}/results/{model}_t5_dir_dc_s*/partition.npz")):
            z = np.load(p)
            blk = pd.Series(z["blocks"], index=z["node_ids"])
            for lname, lser in labelings.items():
                for hemi in ["left", "right"]:
                    ids = side.index[side == hemi]
                    b = blk.reindex(ids).dropna()
                    y = lser.reindex(b.index).dropna()
                    b2 = b.reindex(y.index)
                    if len(y) < MIN_LABELLED or y.nunique() < 2:
                        continue
                    rows.append(dict(model=model, labeling=lname, side=hemi, n=len(y),
                                     n_labels=y.nunique(), n_blocks=b2.nunique(),
                                     AMI=adjusted_mutual_info_score(y.astype(str), b2.astype(int)),
                                     V=v_measure_score(y.astype(str), b2.astype(int))))
            print(f"  {p.split('/')[-2]} done", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(f"{SCRATCH}/t7_which_structure.csv", index=False)
    piv = r.pivot_table(index="labeling", columns="model", values="AMI", aggfunc="mean")
    piv = piv[[m for m in MODELS if m in piv.columns]]
    piv["WINNER"] = piv.idxmax(axis=1)
    meta = r.groupby("labeling").agg(n=("n", "mean"), n_labels=("n_labels", "mean")).round(0)
    print("\n" + "=" * 104)
    print("Adjusted mutual information vs each biological labeling (higher = better recovered)")
    print("chance-corrected, so a model cannot win by producing more blocks")
    print("=" * 104)
    print(piv.join(meta).round(3).to_string())


if __name__ == "__main__":
    main()
