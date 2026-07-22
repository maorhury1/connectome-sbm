"""Verify the two scorer fixes: (1) truncated MLE recovers known params from truncated
samples; (2) stable tail pmf does not underflow to an artificial floor. Run from src/."""
import numpy as np
from scipy.stats import norm
import xval

rng = np.random.default_rng(0)
N, t = 40000, 5
OK = True
def check(name, cond, detail=""):
    global OK; OK = OK and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")

print("== #1 truncated MLE recovers known params (samples truncated at W>=%d) ==" % t)
# lognormal: true mu,sigma on log scale
mu, sig = 2.0, 0.6
w = np.rint(np.exp(rng.normal(mu, sig, N))).astype(int); w = w[w >= t]
p = xval._fit_params("lognormal", w, t)
check("lognormal mu", abs(p["mu"] - mu) < 0.1, f"{p['mu']:.3f} vs {mu}")
check("lognormal sigma", abs(p["sigma"] - sig) < 0.1, f"{p['sigma']:.3f} vs {sig}")
# naive (untruncated) fit is biased -> show it:
naive_mu = np.log(w).mean()
print(f"     (naive mean(log w) = {naive_mu:.3f}: biased HIGH vs true {mu} because W<{t} removed)")

# gaussian
gm, gs = 30.0, 8.0
w = np.rint(rng.normal(gm, gs, N)).astype(int); w = w[w >= t]
p = xval._fit_params("gaussian", w, t)
check("gaussian mu", abs(p["mu"] - gm) < 1.5, f"{p['mu']:.2f} vs {gm}")

# poisson
lam = 12.0
w = rng.poisson(lam, N); w = w[w >= t]
p = xval._fit_params("poisson", w, t)
check("poisson lambda", abs(p["lam"] - lam) < 0.6, f"{p['lam']:.3f} vs {lam}")

# geometric (numpy geometric is k>=1)
pp = 0.1
w = rng.geometric(pp, N); w = w[w >= t]
p = xval._fit_params("geometric", w, t)
check("geometric p", abs(p["p"] - pp) < 0.02, f"{p['p']:.4f} vs {pp}")

print("\n== #2 deep-tail stability (large weight, tight lognormal) ==")
params = {"mu": 1.0, "sigma": 0.3}     # median exp(1)~2.7; k=200 is far in the tail
k = 200
val = xval._log_trunc_pmf("lognormal", k, params, threshold=1)
# naive difference underflows to 0 -> log floor ~ -690:
z = lambda x: norm.cdf((np.log(x) - params["mu"]) / params["sigma"])
naive = np.log(max(z(k + 0.5) - z(k - 0.5), 1e-300))
check("stable value is finite and NOT the artificial floor", np.isfinite(val) and val < -50 and val > -1e6,
      f"stable={val:.1f}")
check("naive underflowed to the floor (shows the bug it fixes)", naive <= -690,
      f"naive={naive:.1f}")
check("stable != naive floor", abs(val - naive) > 1.0, f"gap={val - naive:.1f}")

print("\nRESULT:", "ALL PASS" if OK else "FAILURES ABOVE")
import sys; sys.exit(0 if OK else 1)
