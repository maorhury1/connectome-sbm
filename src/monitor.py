"""
Run monitoring: heartbeat, wall-clock timeout, atomic checkpoints, multi-signal
convergence, and a status board. This is the anti-"stuck-for-weeks" machinery.

It is driven by the fit loop in sbm.py, which calls `Monitor.record(state, S, accept)`
after every `multilevel_mcmc_sweep` — i.e. progress is observed on the REAL production
path, not an opaque wrapper, and the loop can be interrupted between sweeps at any time.
"""
import json
import os
import pickle
import random
import time
import numpy as np
import config


class Timeout(Exception):
    """Raised inside record() when the wall-clock budget is exceeded; the fit loop
    catches it, saves a checkpoint, and marks the run TIMED_OUT (never aggregated as
    if converged)."""


class Monitor:
    def __init__(self, run_name, seed, max_seconds, checkpoint_s=600, heartbeat_s=15,
                 conv_window=10, conv_entropy_tol=1e-4, conv_accept_tol=1e-3):
        self.dir = config.RUNS_DIR / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.progress = self.dir / "progress.jsonl"
        self.state_path = self.dir / "state.pkl"
        self.seed = seed
        self.max_seconds = max_seconds
        self.checkpoint_s = checkpoint_s
        self.heartbeat_s = heartbeat_s
        self.conv = dict(window=conv_window, etol=conv_entropy_tol, atol=conv_accept_tol)
        self.t0 = time.time()
        self._last_hb = 0.0
        self._last_ckpt = self.t0
        self._iter = 0
        self._S_hist = []
        self._acc_hist = []
        self._set_status("RUNNING")

    def record(self, state, S, accept):
        """Call once per sweep. S = running entropy (kept incrementally by the fit loop,
        cheap), accept = n_moves / n_attempts for this sweep."""
        self._iter += 1
        now = time.time()
        elapsed = now - self.t0
        self._S_hist.append(S)
        self._acc_hist.append(accept)
        if now - self._last_hb >= self.heartbeat_s:
            self._heartbeat(state, S, accept, elapsed)
            self._last_hb = now
        if now - self._last_ckpt >= self.checkpoint_s:
            self.save(state, S)
            self._last_ckpt = now
        if elapsed > self.max_seconds:
            self._heartbeat(state, S, accept, elapsed, note="TIMEOUT")
            raise Timeout(f"{self.dir.name}: exceeded {self.max_seconds}s")

    def converged(self):
        """Multi-signal (not entropy alone): running entropy flat AND accept fraction low."""
        w = self.conv["window"]
        if len(self._S_hist) < w + 1:
            return False
        s = self._S_hist[-(w + 1):]
        rel = abs(s[-1] - s[0]) / (abs(s[0]) + 1e-12)
        acc = sum(self._acc_hist[-w:]) / w
        return rel < self.conv["etol"] and acc < self.conv["atol"]

    def save(self, state, S):
        """Atomic checkpoint: temp file + rename. Stores graph checksum, blocks, RNG state."""
        from graph import graph_checksum
        payload = {
            "graph_checksum": graph_checksum(state.g),
            "blocks": _blocks(state),
            "entropy": float(S),
            "seed": self.seed,
            "iter": self._iter,
            "elapsed_s": time.time() - self.t0,
            "rng": {"numpy": np.random.get_state(), "python": random.getstate()},
        }
        tmp = self.state_path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.state_path)   # atomic on the same filesystem

    def finish(self, status, state=None, S=None):
        if state is not None and S is not None:
            self.save(state, S)
        self._set_status(status)

    # ---- internals ----
    def _heartbeat(self, state, S, accept, elapsed, note=""):
        rec = {"t": round(elapsed, 1), "iter": self._iter, "entropy": round(S, 3),
               "n_blocks": _nblocks(state),
               "d_entropy": round(self._S_hist[-1] - self._S_hist[-2], 4) if len(self._S_hist) > 1 else 0.0,
               "accept": round(accept, 4), "seed": self.seed, "note": note}
        with open(self.progress, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _set_status(self, status):
        (self.dir / "STATUS").write_text(status)


def _blocks(state):
    if hasattr(state, "get_bs"):        # NestedBlockState
        return [np.asarray(b.a).copy() for b in state.get_bs()]
    return np.asarray(state.get_blocks().a).copy()


def _nblocks(state):
    b = _blocks(state)
    return int(len(np.unique(b[0] if isinstance(b, list) else b)))


def status():
    """Read-only status board across all runs. Safe to run anytime."""
    rows = []
    for d in sorted(config.RUNS_DIR.glob("*")):
        st = (d / "STATUS").read_text().strip() if (d / "STATUS").exists() else "?"
        last = ""
        p = d / "progress.jsonl"
        if p.exists() and p.read_text().strip():
            r = json.loads(p.read_text().splitlines()[-1])
            last = f"t={r['t']}s iter={r['iter']} blocks={r['n_blocks']} S={r['entropy']:.1f} acc={r['accept']}"
        rows.append(f"{d.name:<44} {st:<11} {last}")
    print("\n".join(rows) if rows else "(no runs yet)")


if __name__ == "__main__":
    status()
