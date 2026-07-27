"""
Report the nested sweep (works on partial results — safe to run mid-sweep).

Reads the per-cell JSONs written by nested_sweep.py and prints one table: description length,
finest-level block count, and the hierarchy shape (blocks per level). Optionally appends the
table to RESULTS.md.

Run:  python report_nested.py --out ~/nested_results [--results-md path/to/RESULTS.md]
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/nested_results"))
    ap.add_argument("--results-md", help="append the table to this RESULTS.md")
    a = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(a.out, "*.json"))):
        r = json.load(open(p))
        r["name"] = os.path.basename(p)[:-5]
        rows.append(r)
    if not rows:
        print(f"no results yet in {a.out}")
        return

    ok = [r for r in rows if r.get("entropy") is not None and r.get("finite")]
    ok.sort(key=lambda r: r["entropy"])                 # lower DL = better compression
    bad = [r for r in rows if r not in ok]

    print(f"NESTED sweep — {len(ok)}/{len(rows)} usable "
          f"(lower entropy = shorter description length)\n")
    print(f"{'model':11} {'dc':4} {'dir':4} {'MDL (M nats)':>13} {'blocks':>7}  hierarchy (blocks/level)")
    print("-" * 84)
    for r in ok:
        print(f"{r['model']:11} {'dc' if r['deg_corr'] else 'ndc':4} "
              f"{'dir' if r['directed'] else 'und':4} {r['entropy']/1e6:13.3f} "
              f"{r['n_blocks']:7d}  {r['levels'][:8]}")
    for r in bad:
        print(f"{r.get('model','?'):11} {'dc' if r.get('deg_corr') else 'ndc':4} "
              f"{'dir' if r.get('directed') else 'und':4} {'FAILED':>13}         "
              f"{r.get('error','non-finite entropy')}")

    if a.results_md:
        md = ["\n\n## Nested SBM sweep (graph-tool >=3.1)\n",
              f"*{len(ok)}/{len(rows)} cells usable. Canonical >=5 graph. "
              "Entropy = description length (lower = better compression). "
              "Hierarchy = blocks per level.*\n",
              "| model | dc | dir | MDL (M nats) | blocks | hierarchy |",
              "|---|---|---|---|---|---|"]
        for r in ok:
            md.append(f"| {r['model']} | {'dc' if r['deg_corr'] else 'ndc'} | "
                      f"{'dir' if r['directed'] else 'und'} | {r['entropy']/1e6:.3f} | "
                      f"{r['n_blocks']} | {r['levels'][:8]} |")
        marker = "\n\n## Nested SBM sweep"
        txt = open(a.results_md).read()
        if marker in txt:
            txt = txt[:txt.index(marker)]
        open(a.results_md, "w").write(txt + "\n".join(md) + "\n")
        print(f"\nappended to {a.results_md}")


if __name__ == "__main__":
    main()
