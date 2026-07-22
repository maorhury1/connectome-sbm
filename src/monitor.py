"""
Watchdog: run a fit as a child process with a hard wall-clock cap (SIGTERM then SIGKILL),
so a run can never hang forever. This is the one piece of run-management we keep.
"""
import subprocess
import signal
import time


def run_supervised(argv, hard_timeout, grace=30):
    """Run `argv` as a child; on wall-clock breach SIGTERM then (after grace) SIGKILL.
    Returns {returncode, killed, elapsed_s}."""
    t0 = time.time()
    proc = subprocess.Popen(argv)
    killed = False
    try:
        proc.wait(timeout=hard_timeout)
    except subprocess.TimeoutExpired:
        killed = True
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return {"returncode": proc.returncode, "killed": killed,
            "elapsed_s": round(time.time() - t0, 1)}
