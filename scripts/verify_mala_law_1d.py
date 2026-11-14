"""Coupled MALA on a 1D standard Gaussian target: verify the rejection-leak theory.

Theory being tested:
  exact 1D Gaussian log-MH-ratio:
    Delta = -(a^2/sqrt2) eta^{3/2} (1-eta a) x xi + (eta^2 a^3/2)(1-eta a/2) x^2 - (eta^2 a^2/2) xi^2
  rejection prob  p = a^{3/2} eta^{3/2}/(sqrt2 pi) + O(eta^2)
  E[xi^2 | reject] -> 2  as eta -> 0
  retained-variance fraction  VR = Var(a'Xbar)_Cstar / Var(a'Xbar)_iid -> 2p  (eta -> 0)
  crude acceptance-independence model: VR = p at every eta.
"""
import numpy as np

a = 1.0  # target N(0, 1/a)

def delta_density(x, y, eta):
    s = 1.0 - eta * a
    return 0.5 * a * (x * x - y * y) + ((y - s * x) ** 2 - (x - s * y) ** 2) / (4.0 * eta)

def delta_poly(x, xi, eta):
    s = 1.0 - eta * a
    return (-(a * a / np.sqrt(2.0)) * eta ** 1.5 * s * x * xi
            + 0.5 * eta ** 2 * a ** 3 * (1.0 - 0.5 * eta * a) * x * x
            - 0.5 * eta ** 2 * a * a * xi * xi)

# ---- validate the exact polynomial form of Delta ----
rng = np.random.default_rng(1)
xt, xit, eta_t = rng.standard_normal(1000), rng.standard_normal(1000), 0.3
yt = (1 - eta_t * a) * xt + np.sqrt(2 * eta_t) * xit
err = np.abs(delta_density(xt, yt, eta_t) - delta_poly(xt, xit, eta_t)).max()
print(f"[check] exact Delta polynomial vs density computation: max abs err = {err:.2e}")

def run(K, eta, E, mode, seed, mh=True):
    """mode in {'cstar','iid'}; returns stationary Var(xbar), time-avg Var, p_rej, E[xi^2|rej]."""
    rng = np.random.default_rng(seed)
    burn = int(15.0 / eta) + 100
    win = int(60.0 / eta) + 500
    s = 1.0 - eta * a
    x = np.zeros((K, E))
    sum_xbar = np.zeros(E)
    var_inst, n_inst = 0.0, 0
    rej_n, rej_xi2, tot = 0, 0.0, 0
    for t in range(burn + win):
        xi = rng.standard_normal((K, E))
        if mode == "cstar":
            xi -= xi.mean(axis=0)
            xi *= np.sqrt(K / (K - 1.0))
        y = s * x + np.sqrt(2.0 * eta) * xi
        if mh:
            D = delta_density(x, y, eta)
            acc = rng.random((K, E)) < np.exp(np.minimum(D, 0.0))
        else:
            acc = np.ones_like(x, dtype=bool)
        xn = np.where(acc, y, x)
        if t >= burn:
            xb = xn.mean(axis=0)
            sum_xbar += xb
            var_inst += xb.var(); n_inst += 1
            r = ~acc
            rej_n += int(r.sum()); rej_xi2 += float((xi[r] ** 2).sum()); tot += K * E
        x = xn
    ta = sum_xbar / win
    p = rej_n / tot
    return var_inst / n_inst, ta.var(ddof=1), p, (rej_xi2 / rej_n if rej_n else float("nan"))

# ---- zero-sum ULA sanity: ensemble mean deterministic ----
v0, _, _, _ = run(8, 0.2, 2000, "cstar", 7, mh=False)
print(f"[check] coupled ULA (no MH) stationary Var(xbar) = {v0:.3e}  (theory: exactly 0)")

K, E = 8, 100_000
print(f"\nK={K}, Gaussian target, ensembles={E}")
hdr = (f"{'eta':>5} | {'p_rej':>7} {'p_th':>7} | {'E[xi2|rej]':>10} | "
       f"{'VR_stat':>8} {'VR_tavg':>8} | {'VR/p':>6} {'VR/2p':>6}")
print(hdr); print("-" * len(hdr))
for i, eta in enumerate((0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2)):
    vs_c, vt_c, p, xi2r = run(K, eta, E, "cstar", 100 + i)
    vs_i, vt_i, _, _ = run(K, eta, E, "iid", 200 + i)
    p_th = a ** 1.5 * eta ** 1.5 / (np.sqrt(2.0) * np.pi)
    VRs, VRt = vs_c / vs_i, vt_c / vt_i
    print(f"{eta:5.2f} | {p:7.4f} {p_th:7.4f} | {xi2r:10.3f} | "
          f"{VRs:8.4f} {VRt:8.4f} | {VRs/p:6.2f} {VRs/(2*p):6.2f}")

# ---- K-independence spot check ----
for K2 in (4, 16):
    vs_c, _, p, _ = run(K2, 0.2, E, "cstar", 300 + K2)
    vs_i, _, _, _ = run(K2, 0.2, E, "iid", 400 + K2)
    print(f"[K-check] K={K2:2d} eta=0.2: VR_stat = {vs_c/vs_i:.4f} (p={p:.4f})")
