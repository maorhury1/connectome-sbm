import numpy as np, data, graph, sbm
pre, post, w = data.load_edges(threshold=1, directed=True)
vals, counts = np.unique(np.concatenate([pre, post]), return_counts=True)
top = vals[np.argsort(counts)[::-1][:6000]]
mask = np.isin(pre, top) & np.isin(post, top)
g, node_ids, _ = graph.build_graph(pre[mask], post[mask], w[mask], directed=True)
state, info = sbm.fit(g, "lognormal", nested=False, seed=0)
print("RESULT:", info, flush=True)
