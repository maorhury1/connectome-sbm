"""
Gate A-1 feasibility probe (v2): can graph-tool 3.0 fit with an edge's WEIGHT masked while
adjacency stays visible, WITHOUT the held-out weight leaking into the partition/params?

v1 was misleading: it used a graph whose partition is driven by adjacency, so weights (and
thus held-out weights) barely affected it -> false "no leak". Here we use a WEIGHT-DEFINED
community structure (adjacency carries no signal; only the weights do), so leakage is
detectable, matching the regime that matters on the real connectome.
"""
import numpy as np
import graph_tool.all as gt
from sklearn.metrics import adjusted_rand_score

# 0. native masking candidate?
doc = (gt.LatentMaskBlockState.__doc__ or "").strip().splitlines()
print("== LatentMaskBlockState (native masking candidate) ==")
print("  " + " ".join(doc[:6]) if doc else "  (no docstring)")

# 1. build a WEIGHT-DEFINED 2-block graph: adjacency = Erdos-Renyi (no group signal);
#    within-group edges heavy, between-group edges light. Only weights reveal the groups.
rng = np.random.default_rng(0)
N, p = 160, 0.15
group = np.array([0] * (N // 2) + [1] * (N // 2))
E = [(i, j) for i in range(N) for j in range(i + 1, N) if rng.random() < p]
E = np.array(E)
same = group[E[:, 0]] == group[E[:, 1]]
w = np.where(same, rng.integers(15, 40, len(E)), rng.integers(1, 6, len(E))).astype(float)

g = gt.Graph(directed=False); g.add_vertex(N); g.add_edge_list(E)
m = g.num_edges()

def fit_partition(weight, edge_subset=None, seed=7):
    if edge_subset is None:
        gg, ww = g, weight
    else:
        gg = gt.Graph(directed=False); gg.add_vertex(N)
        gg.add_edge_list(E[edge_subset]); ww = weight[edge_subset]
    rec = gg.new_ep("double"); rec.a = ww
    gt.seed_rng(seed); np.random.seed(seed)
    st = gt.WeightedBlockState(gg, rec=[rec], rec_types=["real-normal"], deg_corr=True)
    for _ in range(40):
        st.multilevel_mcmc_sweep(niter=1, beta=float("inf"), parallel=True)
    return np.asarray(st.get_blocks().a).copy()

b_full = fit_partition(w)
print(f"\n== sanity: weighted SBM recovers the weight-defined groups? "
      f"ARI(fit, truth) = {adjusted_rand_score(group, b_full):.3f} ==")

held = rng.choice(m, size=m // 5, replace=False)
w_corrupt = w.copy(); w_corrupt[held] = rng.integers(1, 40, len(held)).astype(float)

print("\n== leak test: corrupt held-out weights, keep edges present ==")
ari_leak = adjusted_rand_score(fit_partition(w), fit_partition(w_corrupt))
print(f"  ARI(partition | true , partition | corrupted) = {ari_leak:.3f}")
print("  -> < 1.0 means held-out weight VALUES move the partition == leakage if left in.")

print("\n== leak-free test: remove held-out edges ==")
train = np.setdiff1d(np.arange(m), held)
ari_safe = adjusted_rand_score(fit_partition(w, train), fit_partition(w_corrupt, train))
print(f"  ARI(edges removed, true vs corrupted) = {ari_safe:.3f}  (1.0 = leak-free)")

print("\n== VERDICT ==")
native_ok = (ari_leak > 0.999)
print(f"  native 'keep adjacency, mask weight' leak-free: {native_ok}")
if not native_ok:
    print("  => held-out weights leak into the partition when edges are left in, and gt3 has")
    print("     no per-edge covariate mask. FOLD = edge-removed held-out weight prediction")
    print("     (leak-free; relative weight-model comparison stays fair since removal is common).")
