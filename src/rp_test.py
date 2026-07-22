import graph_tool.all as gt, numpy as np, data, graph
pre, post, w = data.load_edges(threshold=1, directed=True)
vals, counts = np.unique(np.concatenate([pre, post]), return_counts=True)
top = vals[np.argsort(counts)[::-1][:6000]]
mask = np.isin(pre, top) & np.isin(post, top)
g, _, _ = graph.build_graph(pre[mask], post[mask], w[mask], directed=True)
for name, extra in [("default(microcanonical)", {}), ("prior rec_params=[{}]", {"rec_params": [{}]})]:
    gt.seed_rng(0)
    sa = dict(rec=[g.ep["logw"]], rec_types=["real-normal"], deg_corr=True); sa.update(extra)
    st = gt.minimize_blockmodel_dl(g, state=gt.WeightedBlockState, state_args=sa)
    print(f"{name}: nblocks={len(set(st.get_blocks().a))} entropy={st.entropy():.1f}", flush=True)
