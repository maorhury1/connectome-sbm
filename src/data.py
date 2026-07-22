"""
Load the FlyWire codex_2025 connectome and the (evaluation-only) label hierarchy.

Edges: connections_princeton is stored per (pre, post, neuropil). Neuropil is not part
of the model, so we collapse it: one summed syn_count per ordered (pre, post) pair.
Self-loops are dropped. Threshold is applied on the *summed* weight.

Labels are loaded separately and NEVER passed to inference; they exist only for scoring.
"""
import gzip
import numpy as np
import pandas as pd
import config


def load_edges(threshold=1, directed=True):
    """Return (pre, post, weight) int arrays for edges with summed syn_count >= threshold."""
    c = pd.read_csv(config.CONNECTIONS, usecols=["pre_root_id", "post_root_id", "syn_count"])
    c = c[c.pre_root_id != c.post_root_id]                                  # drop self-loops
    e = c.groupby(["pre_root_id", "post_root_id"], sort=False)["syn_count"].sum()
    if not directed:
        # collapse reciprocal pairs onto an unordered key, summing both directions
        a = np.minimum(e.index.get_level_values(0), e.index.get_level_values(1))
        b = np.maximum(e.index.get_level_values(0), e.index.get_level_values(1))
        e = pd.Series(e.values, index=pd.MultiIndex.from_arrays([a, b])).groupby(level=[0, 1]).sum()
    e = e[e >= threshold]
    pre = e.index.get_level_values(0).to_numpy(np.int64)
    post = e.index.get_level_values(1).to_numpy(np.int64)
    w = e.to_numpy(np.int64)
    print(f"[data] edges: {len(w):,} pairs (threshold>={threshold}, directed={directed}), "
          f"weight range [{w.min()},{w.max()}]")
    return pre, post, w


def load_labels():
    """Return a DataFrame indexed by neuron id with the label hierarchy + extra labels.
    Evaluation-only. Untyped neurons keep NaN (handled explicitly at scoring time)."""
    cls = pd.read_csv(config.CLASSIFICATION,
                      usecols=["root_id", "super_class", "class", "sub_class", "side"])
    typ = pd.read_csv(config.CELL_TYPES, usecols=["root_id", "primary_type"])
    nt = pd.read_csv(config.NEURONS, usecols=["root_id", "nt_type"])
    df = cls.merge(typ, on="root_id", how="left").merge(nt, on="root_id", how="left")
    df = df.rename(columns={"root_id": "neuron"}).set_index("neuron")
    n_typed = df["primary_type"].notna().sum()
    print(f"[data] labels: {len(df):,} neurons, {n_typed:,} with primary_type, "
          f"levels={config.LABEL_LEVELS}")
    return df


def data_fingerprint():
    """Small provenance record (file sizes) for versioning; full checksums are slow on netapp."""
    import os
    out = {}
    for p in (config.CONNECTIONS, config.CLASSIFICATION, config.CELL_TYPES, config.NEURONS):
        out[p.name] = os.path.getsize(p)
    return out
