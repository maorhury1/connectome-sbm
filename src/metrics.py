"""
Evaluate a partition against the biology label hierarchy (labels are used ONLY here, never
in fitting). For each label level we report homogeneity, completeness, V-measure, adjusted
mutual information, and adjusted Rand index, over neurons that have both a block and a label.
"""
import numpy as np
from sklearn.metrics import (homogeneity_completeness_v_measure,
                             adjusted_mutual_info_score, adjusted_rand_score)


def evaluate(blocks_by_neuron, labels_df, levels):
    neurons = np.fromiter(blocks_by_neuron.keys(), dtype=np.int64)
    blocks = np.fromiter((blocks_by_neuron[int(n)] for n in neurons), dtype=np.int64)
    out = {}
    for lvl in levels:
        lab = labels_df[lvl].reindex(neurons)
        keep = lab.notna().to_numpy() & (lab.to_numpy().astype(str) != "")
        if keep.sum() < 10:
            continue
        y = lab.to_numpy()[keep].astype(str)
        b = blocks[keep]
        h, c, v = homogeneity_completeness_v_measure(y, b)
        out[lvl] = dict(homogeneity=round(h, 4), completeness=round(c, 4), v_measure=round(v, 4),
                        ami=round(adjusted_mutual_info_score(y, b), 4),
                        ari=round(adjusted_rand_score(y, b), 4),
                        n=int(keep.sum()), n_labels=int(len(set(y))))
    return out
