"""
Build a graph-tool graph from edge arrays, with raw-weight and log-weight edge
properties. Keeps an explicit neuron-id <-> vertex-index mapping (needed to align
partitions with labels) and a cheap content checksum (for checkpoint validation).
"""
import hashlib
import numpy as np
import graph_tool.all as gt


def build_graph(pre, post, weight, directed=True):
    """Return (g, node_ids, idx_of) with edge properties g.ep.w (raw) and g.ep.logw (log).
    node_ids[v] = neuron id of vertex v; idx_of[neuron] = v."""
    node_ids = np.unique(np.concatenate([pre, post]))
    idx_of = {int(n): i for i, n in enumerate(node_ids)}
    src = np.fromiter((idx_of[int(x)] for x in pre), dtype=np.int64, count=len(pre))
    dst = np.fromiter((idx_of[int(x)] for x in post), dtype=np.int64, count=len(post))

    g = gt.Graph(directed=directed)
    g.add_vertex(len(node_ids))
    g.add_edge_list(np.column_stack([src, dst]))

    w = g.new_ep("double"); w.a = weight.astype(np.float64)
    logw = g.new_ep("double"); logw.a = np.log(weight.astype(np.float64))
    g.ep["w"] = w
    g.ep["logw"] = logw
    print(f"[graph] {g.num_vertices():,} vertices, {g.num_edges():,} edges, directed={directed}")
    return g, node_ids, idx_of


def graph_checksum(g):
    """Cheap order-invariant checksum of (structure + weights) for checkpoint validation."""
    h = hashlib.sha1()
    h.update(np.asarray(g.get_edges()).tobytes())
    h.update(np.asarray(g.ep["w"].a).tobytes())
    h.update(str(g.is_directed()).encode())
    return h.hexdigest()[:16]
