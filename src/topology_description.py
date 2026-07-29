"""Topology + partition description length for a FIXED candidate partition (CPWDL term S_A).

Rebuilds an UNWEIGHTED degree-corrected BlockState with the candidate partition held fixed and
returns its entropy: adjacency + degree-corrected topology + graph-tool's partition prior.

Computed fresh -- never by subtracting weight terms from a weighted entropy. Identical across
the five weight families for the same partition, so it is cached per partition.
"""
import numpy as np


def topology_dl(g, blocks, deg_corr=True):
    """graph-tool unweighted entropy with b fixed to `blocks` (array over vertex index)."""
    import graph_tool.all as gt
    b = g.new_vp("int")
    b.a = np.asarray(blocks, dtype=np.int64)
    state = gt.BlockState(g, b=b, deg_corr=deg_corr)
    return float(state.entropy())
