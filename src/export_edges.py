"""
Export the canonical edge list to a compact, portable .npz.

Why: the nested fits require graph-tool >=3.1, which cannot run on this server (its binaries
need glibc 2.38; the machine has 2.35 — see PLAN.md 3.4 clarification). So the nested sweep is
run on another machine, and this writes the one file it needs (~24 MB) instead of copying the
whole dataset.

Contents: src/dst as int32 node INDICES, w as int32 synapse counts, and node_ids (int64) mapping
index -> FlyWire root_id so partitions can be mapped back to neurons.

Run from src/:  python export_edges.py [--threshold 5] [--out PATH]
"""
import argparse
import numpy as np
import config
import data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--out", default=str(config.WORK_DIR / "edges_t5_dir.npz"))
    a = ap.parse_args()

    # directed edge list is canonical; the sweep collapses it for undirected runs
    pre, post, w = data.load_edges(threshold=a.threshold, directed=True)
    ids = np.unique(np.concatenate([pre, post]))
    idx = {int(v): i for i, v in enumerate(ids)}
    src = np.fromiter((idx[int(x)] for x in pre), dtype=np.int32, count=len(pre))
    dst = np.fromiter((idx[int(x)] for x in post), dtype=np.int32, count=len(post))
    np.savez_compressed(a.out, src=src, dst=dst, w=w.astype(np.int32),
                        node_ids=ids.astype(np.int64))
    import os
    print(f"[export] nodes={len(ids):,} edges={len(src):,} threshold>={a.threshold} "
          f"-> {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
