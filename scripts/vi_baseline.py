"""Full-rank Gaussian VI (and EP) baseline for the exact-gold end-task comparison.

Baseline check: is a full-rank Gaussian approximation already a
good posterior-mean estimate on these tempered-likelihood logistic posteriors,
making sampling unnecessary?  Same potential and data pipeline as
scripts/map_baseline.py and scripts/rr_extrapolation.py:

    U(theta) = mean_i [softplus(x_i . theta) - y_i (x_i . theta)]
               + (prior/2) ||theta * mask||^2,   mask[0] = 0 (intercept free),

target density pi ~ exp(-U).  VI fits q = N(mu, L L^T) by minimizing
KL(q||pi), i.e. the negative ELBO E_q[U] - H[q], deterministically (no
Monte Carlo), so the reported point is the reverse-KL (ELBO) family
optimum rather than one stochastic ADVI run's output.

Quadrature note: these tempered posteriors are WIDE (marginal scales
s_i = ||L^T x_i|| up to ~50), and softplus has poles at Im(z) = pi, so
Gauss--Hermite in the standardized variable converges at rate exp(-c
sqrt(Q) pi / s) -- uselessly slow at large s (measured: the VI mean still
drifts 1e-4 between Q=61 and Q=241).  Instead we use the exact hinge
decomposition softplus(z) = max(z, 0) + log1p(exp(-|z|)):
E[max(Z,0)] = m Phi(m/s) + s phi(m/s) in closed form, and the bounded
remainder r(z) = log1p(exp(-|z|)) <= e^{-|z|} integrated against
N(z; m, s^2) on FIXED z-space Gauss--Legendre panels [-46,0] and [0,46]
(panel split at r's derivative kink; poles stay pi from the axis, so the
GL rate is s-independent; truncation error <= e^{-46}).  Value and
gradient are exact derivatives of the same discretization, so L-BFGS sees
a consistent objective; a doubling gate (--panel-nodes) verifies the
quadrature is converged to machine precision.  The negative ELBO is
jointly convex in (mu, L) for log-concave targets (Challis & Barber 2013);
we converge from two inits (Laplace and a cold start) and require
agreement.  EP (best effort): damped parallel EP with per-site tempered
factors exp(-(softplus(z_i)-y_i z_i)/n) on z_i = x_i . theta, Gaussian
prior factor exact, tilted moments by the same fixed z-panels plus
closed-form truncated-Gaussian tails for |z| > 46 (where the remainder
factor is 1 to double precision), initialized at the Laplace site
decomposition (whose global mean is exactly the MAP).

Per cell reports ||mu - theta_gold||^2 against the pooled gold posterior
mean, the ratio to the zero-sum ULA@4eta_c single-level median MSE, and
trace of the fitted covariance.  Deterministic -- no seeds.  Writes
outputs/sweeps/vi_baseline.jsonl (refuses to overwrite) and prints the
'{:.3g}'-formatted VI/EP columns for paper/tables/gold_endtask.tex (this
script does not edit the table).
"""
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.optimize import minimize
from scipy.special import ndtr

DATASETS = ["wdbc", "spambase", "adult"]
PRIORS = [0.1, 0.3]
DATASET_LABEL = {"wdbc": "WDBC", "spambase": "Spambase", "adult": "Adult"}
GL_PANEL_NODES = 320
Z_CUT = 46.0
CHUNK_ROWS = 8192
SQRT_2PI = np.sqrt(2.0 * np.pi)


def gl_panels(n_panel):
    """Fixed Gauss-Legendre nodes/weights on [-Z_CUT, 0] and [0, Z_CUT]."""
    t, w = leggauss(n_panel)
    half = Z_CUT / 2.0
    z_neg = half * (t - 1.0)
    z_pos = half * (t + 1.0)
    z = np.concatenate([z_neg, z_pos])
    ww = np.concatenate([w, w]) * half
    return z, ww


def norm_pdf(t):
    return np.exp(-0.5 * t * t) / SQRT_2PI


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def file_sha256(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except Exception:
        return "missing"


def provenance(args):
    import scipy
    return {
        "argv": " ".join(sys.argv),
        "script_sha256": file_sha256(__file__),
        "precommit_sha256": file_sha256("campaigns/vi_ep_baseline_precommit.md"),
        "gold_sha256": file_sha256(args.gold),
        "rr_sha256": [file_sha256(p) for p in args.rr],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "gtol": args.gtol,
        "maxiter": args.maxiter,
        "panel_nodes": args.panel_nodes,
    }


def add_intercept(x_np):
    ones = np.ones((x_np.shape[0], 1), dtype=np.float32)
    return np.concatenate([ones, x_np.astype(np.float32)], axis=1)


def load_dataset(data_dir, name):
    data = np.load(data_dir / f"{name}.npz", allow_pickle=True)
    x = add_intercept(data["X_train"]).astype(np.float64)
    y = data["y_train"].astype(np.float32).astype(np.float64)
    return x, y


def sigmoid_stable(z):
    p = np.empty_like(z)
    pos = z >= 0
    p[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    p[~pos] = ez / (1.0 + ez)
    return p


def softplus_stable(z):
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def grad_hess(theta, x, y, prior, mask):
    n = x.shape[0]
    z = x @ theta
    p = sigmoid_stable(z)
    grad = x.T @ (p - y) / n + prior * theta * mask
    w = p * (1.0 - p)
    hess = (x.T * w) @ x / n + prior * np.diag(mask)
    return grad, hess


def potential(theta, x, y, prior, mask):
    z = x @ theta
    u = np.mean(softplus_stable(z) - y * z)
    return u + 0.5 * prior * np.sum((theta * mask) ** 2)


def newton_map(x, y, prior, tol=1e-10, max_iter=200):
    d = x.shape[1]
    mask = np.ones(d)
    mask[0] = 0.0
    theta = np.zeros(d)
    u = potential(theta, x, y, prior, mask)
    for it in range(max_iter):
        grad, hess = grad_hess(theta, x, y, prior, mask)
        gnorm = float(np.linalg.norm(grad))
        if gnorm < tol:
            return theta, hess
        step = cho_solve(cho_factor(hess, lower=True), grad)
        t = 1.0
        while t > 1e-8:
            cand = theta - t * step
            u_cand = potential(cand, x, y, prior, mask)
            if u_cand <= u - 1e-4 * t * float(grad @ step):
                theta, u = cand, u_cand
                break
            t *= 0.5
        else:
            raise RuntimeError("line search failed")
    raise RuntimeError(f"Newton did not converge (gnorm={gnorm:.3e})")


class PackedVI:
    """negELBO(mu, L) with L = tril; diagonal via log-parameterization.

    E[softplus] per data point = closed-form hinge + fixed-z-panel GL
    quadrature of the remainder; B = dE/dm and C = dE/ds are the exact
    derivatives of the same discretization.
    """

    def __init__(self, x, y, prior, mask, n_panel=GL_PANEL_NODES):
        self.x, self.y, self.prior, self.mask = x, y, prior, mask
        self.n, self.d = x.shape
        zk, wk = gl_panels(n_panel)
        self.zk = zk
        self.wr = wk * np.log1p(np.exp(-np.abs(zk)))
        d = self.d
        self.tril_r, self.tril_c = np.tril_indices(d, k=-1)
        self.n_off = len(self.tril_r)
        self.min_s_seen = np.inf

    def pack(self, mu, L):
        return np.concatenate([mu, np.log(np.diag(L)), L[self.tril_r, self.tril_c]])

    def unpack(self, p):
        d = self.d
        mu = p[:d]
        L = np.zeros((d, d))
        L[np.diag_indices(d)] = np.exp(p[d:2 * d])
        L[self.tril_r, self.tril_c] = p[2 * d:]
        return mu, L

    def _esp_b_c(self, m, s):
        """E[softplus(N(m, s^2))], its dm- and ds-derivatives (chunk of rows)."""
        t = m / s
        pdf_t = norm_pdf(t)
        cdf_t = ndtr(t)
        e_sp = m * cdf_t + s * pdf_t
        b = cdf_t.copy()
        c = pdf_t.copy()
        dz = self.zk[None, :] - m[:, None]
        s2 = (s * s)[:, None]
        gauss = np.exp(-0.5 * dz * dz / s2) / (s[:, None] * SQRT_2PI)
        e_sp += gauss @ self.wr
        b += ((gauss * dz) @ self.wr) / (s * s)
        c += ((gauss * dz * dz) @ self.wr) / (s ** 3) - (gauss @ self.wr) / s
        return e_sp, b, c

    def neg_elbo_grad(self, p):
        x, y, prior, mask = self.x, self.y, self.prior, self.mask
        n, d = self.n, self.d
        mu, L = self.unpack(p)
        like_sum = 0.0
        gmu_data = np.zeros(d)
        gl_data = np.zeros((d, d))
        min_s = np.inf
        for lo in range(0, n, CHUNK_ROWS):
            xc = x[lo:lo + CHUNK_ROWS]
            yc = y[lo:lo + CHUNK_ROWS]
            m = xc @ mu
            a = xc @ L
            s = np.sqrt(np.sum(a * a, axis=1))
            min_s = min(min_s, float(s.min()))
            if not np.isfinite(s).all() or s.min() <= 1e-8:
                raise RuntimeError(f"marginal scale collapsed or non-finite: min s = {s.min()}")
            e_sp, b, c = self._esp_b_c(m, s)
            like_sum += float(np.sum(e_sp - yc * m))
            gmu_data += xc.T @ (b - yc)
            gl_data += xc.T @ ((c / s)[:, None] * a)
        self.min_s_seen = min(self.min_s_seen, min_s)
        like = like_sum / n
        row_var = np.sum(L * L, axis=1)
        prior_term = 0.5 * prior * float(np.sum(mask * (mu * mu + row_var)))
        entropy = 0.5 * d * np.log(2.0 * np.pi * np.e) + float(np.sum(p[d:2 * d]))
        val = like + prior_term - entropy

        gmu = gmu_data / n + prior * mask * mu
        gl = gl_data / n + prior * mask[:, None] * L
        gdiag = np.diag(gl) * np.diag(L) - 1.0
        goff = gl[self.tril_r, self.tril_c]
        return val, np.concatenate([gmu, gdiag, goff])


def fit_vi(x, y, prior, mask, theta_map, hess, gtol, maxiter,
           n_panel=GL_PANEL_NODES, init_names=("laplace", "cold")):
    prob = PackedVI(x, y, prior, mask, n_panel=n_panel)
    sigma_lap = cho_solve(cho_factor(hess, lower=True), np.eye(hess.shape[0]))
    inits = {
        "laplace": (theta_map.copy(), cholesky(sigma_lap, lower=True)),
        # wide enough that every data-point marginal stays panel-resolved
        "cold": (np.zeros(prob.d), 0.5 * np.eye(prob.d)),
    }
    sols = {}
    for name in init_names:
        mu0, l0 = inits[name]
        res = minimize(prob.neg_elbo_grad, prob.pack(mu0, l0), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=maxiter, maxfun=4 * maxiter, ftol=1e-15, gtol=gtol))
        val, grad = prob.neg_elbo_grad(res.x)
        mu, L = prob.unpack(res.x)
        sols[name] = dict(mu=mu, L=L, neg_elbo=float(val),
                          grad_inf=float(np.max(np.abs(grad))), n_iter=int(res.nit),
                          success=bool(res.success), message=str(res.message))
    best = min(sols.values(), key=lambda sv: sv["neg_elbo"])
    agree = (float(np.sum((sols["laplace"]["mu"] - sols["cold"]["mu"]) ** 2))
             if len(sols) > 1 else float("nan"))
    a_best = x @ best["L"]
    min_s_final = float(np.sqrt(np.sum(a_best * a_best, axis=1)).min())
    return best, agree, sols, prob.min_s_seen, min_s_final


VI_GATES = dict(grad_inf=1e-6, agree_sq=1e-8, quad_drift_sq=1e-10, min_s_final=1.0)


def enforce_vi_gates(cell, sols, agree, quad_drift, min_s_final):
    """Abort (exit nonzero) BEFORE any gold scoring if a verification gate fails."""
    failures = []
    for name, sv in sols.items():
        if not sv["success"]:
            failures.append(f"optimizer[{name}] did not converge: {sv['message']}")
        if not np.isfinite(sv["mu"]).all() or not np.isfinite(sv["neg_elbo"]):
            failures.append(f"non-finite solution[{name}]")
        if sv["grad_inf"] >= VI_GATES["grad_inf"]:
            failures.append(f"grad_inf[{name}] {sv['grad_inf']:.3e} >= {VI_GATES['grad_inf']}")
    if not (agree < VI_GATES["agree_sq"]):
        failures.append(f"init agreement {agree:.3e} >= {VI_GATES['agree_sq']}")
    if not (quad_drift < VI_GATES["quad_drift_sq"]):
        failures.append(f"quadrature drift {quad_drift:.3e} >= {VI_GATES['quad_drift_sq']}")
    if not (min_s_final >= VI_GATES["min_s_final"]):
        failures.append(f"min final marginal scale {min_s_final:.3g} < {VI_GATES['min_s_final']}")
    if failures:
        raise SystemExit(f"VI GATE FAILURE at cell {cell}: " + "; ".join(failures))


def check_grad(x, y, prior):
    d = x.shape[1]
    mask = np.ones(d)
    mask[0] = 0.0
    theta_map, hess = newton_map(x, y, prior)
    prob = PackedVI(x, y, prior, mask)
    sigma_lap = cho_solve(cho_factor(hess, lower=True), np.eye(d))
    l0 = cholesky(sigma_lap, lower=True)
    rng = np.random.default_rng(20260715)
    p0 = prob.pack(theta_map, l0) + 0.01 * rng.standard_normal(2 * d + prob.n_off)
    _, g = prob.neg_elbo_grad(p0)
    idx = rng.choice(len(p0), size=min(40, len(p0)), replace=False)
    # eps at the central-difference optimum ~cbrt(u*|f|); per-coordinate floor
    # is then ~1e-6 relative on small-|g| coords, so gate at 1e-4.
    eps = 2e-5
    worst = 0.0
    for i in idx:
        pp, pm = p0.copy(), p0.copy()
        pp[i] += eps
        pm[i] -= eps
        fd = (prob.neg_elbo_grad(pp)[0] - prob.neg_elbo_grad(pm)[0]) / (2 * eps)
        rel = abs(fd - g[i]) / max(abs(fd), abs(g[i]), 1e-12)
        worst = max(worst, rel)
    print(f"gradient check: worst per-coord rel err {worst:.3e} over {len(idx)} coords")
    worst_dir = 0.0
    for _ in range(5):
        v = rng.standard_normal(len(p0))
        v /= np.linalg.norm(v)
        fd = (prob.neg_elbo_grad(p0 + eps * v)[0] - prob.neg_elbo_grad(p0 - eps * v)[0]) / (2 * eps)
        an = float(g @ v)
        worst_dir = max(worst_dir, abs(fd - an) / max(abs(fd), abs(an), 1e-12))
    print(f"gradient check: worst directional rel err {worst_dir:.3e} over 5 directions")
    if worst > 1e-4 or worst_dir > 1e-6:
        raise SystemExit("gradient check FAILED")


def tilted_moments(m_cav, v_cav, y, n, zk, wk, sp_k):
    """Centered moments D_p = E_{N(m,v)}[(z-m)^p f(z)], p=0,1,2, per site,
    with f(z) = exp(-(softplus(z)-y z)/n); returns (mt, vt).

    Core: fixed z-panel GL on [-Z_CUT, Z_CUT].  Tails |z| > Z_CUT: the
    softplus remainder log1p(e^{-|z|}) is < e^{-Z_CUT}, so the factor is
    exp(a z) exactly (to double precision) with a linear exponent, giving
    closed-form truncated-Gaussian tails.  Centering about the cavity mean
    avoids the E[z^2]-m^2 cancellation at large |m|.
    """
    sd = np.sqrt(v_cav)
    delta = zk[None, :] - m_cav[:, None]
    logi = (-0.5 * delta * delta / v_cav[:, None]
            - np.log(sd[:, None] * SQRT_2PI)
            - (sp_k[None, :] - y[:, None] * zk[None, :]) / n)
    core = np.exp(logi)
    d0 = (core * wk[None, :]).sum(axis=1)
    d1 = (core * (wk[None, :] * delta)).sum(axis=1)
    d2 = (core * (wk[None, :] * delta * delta)).sum(axis=1)

    def right_tail_centered(m, v, a, c):
        # int_{z>c} (z-m)^p N(z;m,v) e^{az} dz, p=0,1,2 (centered at m).
        sdl = np.sqrt(v)
        f = np.exp(a * m + 0.5 * a * a * v)
        mp_shift = a * v
        alpha = (c - m - mp_shift) / sdl
        q = ndtr(-alpha)
        pdf = norm_pdf(alpha)
        e1 = sdl * pdf + mp_shift * q
        e2 = v * (q + alpha * pdf) + 2.0 * mp_shift * sdl * pdf + mp_shift * mp_shift * q
        return f * q, f * e1, f * e2

    a_r = -(1.0 - y) / n
    t0r, t1r, t2r = right_tail_centered(m_cav, v_cav, a_r, Z_CUT)
    # Left tail via reflection z -> -z: (m, a) -> (-m, -a), (z-m)^p -> (-1)^p.
    a_l = y / n
    t0l, t1l, t2l = right_tail_centered(-m_cav, v_cav, -a_l, Z_CUT)
    d0 = d0 + t0r + t0l
    d1 = d1 + t1r - t1l
    d2 = d2 + t2r + t2l
    r1 = d1 / d0
    mt = m_cav + r1
    vt = d2 / d0 - r1 * r1
    return mt, vt


def check_ep_moments():
    """Factor == 1 (n -> inf) must return the cavity moments exactly."""
    zk, wk = gl_panels(GL_PANEL_NODES)
    sp_k = softplus_stable(zk)
    cases = [(0.0, 1.0), (10.0, 40.0), (-5.0, 0.7), (17.0, 47.0), (-30.0, 3.0),
             (3.0, 400.0), (-2.0, 550.0)]
    worst = 0.0
    for m, sd in cases:
        mt, vt = tilted_moments(np.array([m]), np.array([sd * sd]), np.array([1.0]),
                                np.inf, zk, wk, sp_k)
        worst = max(worst, abs(mt[0] - m) / max(abs(m), 1.0),
                    abs(vt[0] - sd * sd) / (sd * sd))
    print(f"EP moment identity check (factor=1): worst rel err {worst:.3e}")
    if worst > 1e-12:
        raise SystemExit("EP moment identity check FAILED")


def check_quad():
    """Gold-independent brute-force validation of both quadrature kernels
    against adaptive scipy.integrate.quad, over a stress grid that covers
    the measured scale range (s up to ~460, cavity variance up to ~3e5)."""
    from scipy.integrate import quad

    zk, wk = gl_panels(GL_PANEL_NODES)
    sp_k = softplus_stable(zk)
    prob_like = PackedVI(np.ones((1, 1)), np.zeros(1), 0.0, np.zeros(1))

    def r_f(z):
        return np.log1p(np.exp(-abs(z)))

    def sig_r(z):
        return sigmoid_stable(np.array([z]))[0] - (1.0 if z > 0 else 0.0)

    def mixed_err(val, ref):
        # A component with absolute error < 1e-14 cannot move gradients
        # gated at 1e-6; treat such agreement as exact.
        if abs(val - ref) < 1e-14:
            return 0.0
        return abs(val - ref) / max(abs(ref), 1e-8)

    ms = [-30.0, -5.0, 0.0, 3.0, 17.0]
    ss = [0.6, 1.0, 5.0, 47.0, 460.0]
    worst_v = worst_b = worst_c = 0.0
    for m in ms:
        for s in ss:
            def gauss(z):
                return np.exp(-0.5 * (z - m) ** 2 / (s * s)) / (s * SQRT_2PI)

            ref_rem = sum(quad(lambda z: r_f(z) * gauss(z), a, b, epsabs=1e-15, limit=400)[0]
                          for a, b in [(-Z_CUT, 0.0), (0.0, Z_CUT)])
            ref_b_rem = sum(quad(lambda z: sig_r(z) * gauss(z), a, b, epsabs=1e-15, limit=400)[0]
                            for a, b in [(-Z_CUT, 0.0), (0.0, Z_CUT)])
            ref_c_rem = sum(quad(lambda z: sig_r(z) * gauss(z) * (z - m) / s, a, b,
                                 epsabs=1e-15, limit=400)[0]
                            for a, b in [(-Z_CUT, 0.0), (0.0, Z_CUT)])
            t = m / s
            ref_v = m * ndtr(t) + s * norm_pdf(t) + ref_rem
            ref_b = ndtr(t) + ref_b_rem
            ref_c = norm_pdf(t) + ref_c_rem
            e_sp, b, c = prob_like._esp_b_c(np.array([m]), np.array([s]))
            worst_v = max(worst_v, mixed_err(e_sp[0], ref_v))
            worst_b = max(worst_b, mixed_err(b[0], ref_b))
            worst_c = max(worst_c, mixed_err(c[0], ref_c))
    print(f"quad check E[softplus]/B/C vs adaptive quad: worst rel "
          f"{worst_v:.3e} / {worst_b:.3e} / {worst_c:.3e}")
    if max(worst_v, worst_b, worst_c) > 1e-10:
        raise SystemExit("VI quadrature check FAILED")

    # The centered first moment cancels between the delta<0 and delta>0
    # halves, so BOTH integrators carry roundoff proportional to the L1
    # mass of the integrand (scipy warns its error is underestimated
    # there).  Errors are therefore normalized by the L1 mass, computed
    # by quad on the positive integrand |delta|^p N f (no cancellation).
    import warnings
    from scipy.integrate import IntegrationWarning

    worst_mt = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        for m in [-15.0, -3.0, 0.0, 8.0]:
            for v in [0.36, 25.0, 2.2e3, 3.0e5]:
                for y_val in [0.0, 1.0]:
                    for n_val in [455.0, 30162.0]:
                        sd = np.sqrt(v)

                        def integrand(z, p, absolute=False):
                            gz = np.exp(-0.5 * (z - m) ** 2 / v) / (sd * SQRT_2PI)
                            fz = np.exp(-(r_f(z) + max(z, 0.0) - y_val * z) / n_val)
                            base = (z - m) ** p
                            return (abs(base) if absolute else base) * gz * fz

                        for p in range(3):
                            ref = sum(quad(integrand, a, b, args=(p,), epsabs=1e-16, limit=400)[0]
                                      for a, b in [(-Z_CUT, 0.0), (0.0, Z_CUT)])
                            l1 = sum(quad(integrand, a, b, args=(p, True), epsabs=1e-16,
                                          limit=400)[0]
                                     for a, b in [(-Z_CUT, 0.0), (0.0, Z_CUT)])
                            panel = (np.exp(-0.5 * (zk - m) ** 2 / v) / (sd * SQRT_2PI)
                                     * np.exp(-(sp_k - y_val * zk) / n_val)
                                     * wk * (zk - m) ** p).sum()
                            worst_mt = max(worst_mt, abs(panel - ref) / max(l1, 1e-300))
    print(f"quad check EP tilted core vs adaptive quad (L1-normalized): worst {worst_mt:.3e}")
    if worst_mt > 1e-9:
        raise SystemExit("EP tilted-moment quadrature check FAILED")


def fit_ep(x, y, prior, mask, theta_map, dampings=(0.5, 0.3, 0.15), max_sweeps=5000,
           tol=1e-8, n_panel=GL_PANEL_NODES):
    """Damped parallel EP on z-space sites; returns (result dict, converged).

    Convergence (strict, all required at the SAME sweep): relative
    site-parameter residual < tol; zero skipped cavities; all tilted
    variances finite and positive; min site precision > -1e-10 (exact
    log-concave tilted matching gives tau >= 0; material negativity means
    a moment-accuracy failure, not a fixed point); global precision PD
    (Cholesky); mean stability max|mu - mu_prev| < 1e-10.  Damping factors
    are applied fractions of the update, so SMALLER is more damped:
    fallbacks 0.5 -> 0.3 -> 0.15.
    """
    n, d = x.shape
    zk, wk = gl_panels(n_panel)
    sp_k = softplus_stable(zk)
    p0_diag = prior * mask
    z_map = x @ theta_map
    p_map = sigmoid_stable(z_map)
    w_map = p_map * (1.0 - p_map)
    last = dict(sweeps=0, damping=dampings[0], delta=float("inf"), reason="did not run")

    for damping in dampings:
        tau = w_map / n
        nu = (w_map * z_map - (p_map - y)) / n
        mu_prev = None
        chol_fail = False
        for sweep in range(1, max_sweeps + 1):
            lam = np.diag(p0_diag) + (x.T * tau) @ x
            try:
                cf = cho_factor(lam, lower=True)
            except np.linalg.LinAlgError:
                chol_fail = True
                break
            sigma = cho_solve(cf, np.eye(d))
            mu = sigma @ (x.T @ nu)
            xs = x @ sigma
            s2 = np.sum(xs * x, axis=1)
            m = x @ mu
            prec_cav = 1.0 / s2 - tau
            # cavity must exist and stay panel-resolved (sd <= is fine; the
            # panels under-resolve only below sd ~ 0.5, i.e. prec_cav > 4)
            good = (prec_cav > 1e-12) & (prec_cav < 4.0)
            v_cav = np.where(good, 1.0 / np.maximum(prec_cav, 1e-12), 1.0)
            m_cav = np.where(good, (m / s2 - nu) * v_cav, m)
            mt = np.empty(n)
            vt = np.empty(n)
            for lo in range(0, n, CHUNK_ROWS):
                sl = slice(lo, min(lo + CHUNK_ROWS, n))
                mt[sl], vt[sl] = tilted_moments(m_cav[sl], v_cav[sl], y[sl], n, zk, wk, sp_k)
            vt_ok = np.isfinite(vt) & (vt > 0)
            good = good & vt_ok
            tau_new = 1.0 / vt - prec_cav
            nu_new = mt / vt - m_cav * prec_cav
            d_tau = np.where(good, tau_new - tau, 0.0)
            d_nu = np.where(good, nu_new - nu, 0.0)
            tau = tau + damping * d_tau
            nu = nu + damping * d_nu
            scale_t = max(np.max(np.abs(tau)), 1e-30)
            scale_n = max(np.max(np.abs(nu)), 1e-30)
            delta = max(np.max(np.abs(d_tau)) / scale_t, np.max(np.abs(d_nu)) / scale_n)
            n_skipped = int(np.sum(~good))
            mu_step = float(np.max(np.abs(mu - mu_prev))) if mu_prev is not None else float("inf")
            mu_prev = mu
            last = dict(sweeps=sweep, damping=damping, delta=float(delta),
                        n_skipped=n_skipped, mu_step=mu_step,
                        min_tau=float(tau.min()), reason="max sweeps")
            if (delta < tol and n_skipped == 0 and tau.min() > -1e-10
                    and mu_step < 1e-10):
                lam = np.diag(p0_diag) + (x.T * tau) @ x
                sigma = cho_solve(cho_factor(lam, lower=True), np.eye(d))
                mu = sigma @ (x.T @ nu)
                return dict(mu=mu, trace_sigma=float(np.trace(sigma)), sweeps=sweep,
                            damping=damping, delta=float(delta), n_skipped=0,
                            min_tau=float(tau.min()), mu_step=mu_step), True
        if chol_fail:
            last["reason"] = "cholesky failure"
            continue
    last["mu"] = None
    return last, False


def load_gold(path):
    gold = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("kind") == "gold":
                gold[(row["dataset"], row["prior"])] = row
    return gold


def load_gold_checked(path, expected_schema_version, expected_gold_ids):
    """P4-era gold loader (screens_precommit.md Amendment v3, V3-5).

    expected_gold_ids: {(dataset, prior): gold_id}.  Every gold row in the
    file must carry expected_schema_version; duplicate (dataset, prior,
    gold_id) rows ABORT (never last-win); each expected cell must resolve to
    exactly one row with its registered gold_id (rows for other arms/attempts
    of the same cell may coexist and are simply not selected).  Returns
    {(dataset, prior): row} for exactly the expected cells."""
    rows = {}
    counts = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") != "gold":
            continue
        sv = row.get("schema_version")
        if sv != expected_schema_version:
            raise SystemExit(f"gold loader: {path}: schema_version {sv!r} != "
                             f"expected {expected_schema_version!r}")
        key = (row["dataset"], row["prior"], row["gold_id"])
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > 1:
            raise SystemExit(f"gold loader: {path}: duplicate gold row for cell "
                             f"{key[:2]} gold_id={key[2]!r} (refusing last-win)")
        rows[key] = row
    out = {}
    for cell, gid in expected_gold_ids.items():
        key = (cell[0], cell[1], gid)
        if key not in rows:
            raise SystemExit(f"gold loader: {path}: no gold row for cell {cell} "
                             f"with expected gold_id {gid!r}")
        out[cell] = rows[key]
    return out


def load_zs_medians(paths):
    per_cell = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("kind") == "estimator" and r.get("estimator") == "single"
                    and r.get("coupling") == "zero_sum" and r.get("level") == "eta1"
                    and r.get("pair") == "primary"):
                per_cell.setdefault((r["dataset"], r["prior"]), []).append(r["mse_gold_median"])
    return per_cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--gold", type=Path, default=Path("outputs/sweeps/rr_extrapolation_gold.jsonl"))
    parser.add_argument("--rr", nargs="+", type=Path,
                        default=[Path(f"outputs/sweeps/rr_extrapolation_rep{r}.jsonl") for r in range(3)])
    parser.add_argument("--out", type=Path, default=Path("outputs/sweeps/vi_baseline.jsonl"))
    parser.add_argument("--gtol", type=float, default=1e-9)
    parser.add_argument("--maxiter", type=int, default=20000)
    parser.add_argument("--panel-nodes", type=int, default=GL_PANEL_NODES)
    parser.add_argument("--quick", action="store_true", help="wdbc only, loose tolerances")
    parser.add_argument("--check-grad", action="store_true")
    parser.add_argument("--check-quad", action="store_true")
    parser.add_argument("--skip-ep", action="store_true")
    args = parser.parse_args()

    if args.check_quad:
        check_ep_moments()
        check_quad()
        return

    if args.check_grad:
        check_ep_moments()
        x, y = load_dataset(args.data_dir, "wdbc")
        check_grad(x, y, 0.1)
        return

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    datasets = ["wdbc"] if args.quick else DATASETS
    gtol = 1e-7 if args.quick else args.gtol
    max_sweeps = 200 if args.quick else 5000

    prov = provenance(args)
    gold = load_gold(args.gold)
    zs = load_zs_medians(args.rr)
    rows = []
    ep_all_converged = True
    for ds in datasets:
        x, y = load_dataset(args.data_dir, ds)
        d = x.shape[1]
        mask = np.ones(d)
        mask[0] = 0.0
        for prior in PRIORS:
            g = gold[(ds, prior)]
            assert d == g["dim"], (ds, prior, x.shape, g["dim"])
            zs_reps = zs[(ds, prior)]
            zs_med = float(statistics.median(zs_reps))
            theta_map, hess = newton_map(x, y, prior)

            best, agree, sols, min_s_seen, min_s_final = fit_vi(
                x, y, prior, mask, theta_map, hess, gtol, args.maxiter,
                n_panel=args.panel_nodes)
            fine, _, _, _, _ = fit_vi(x, y, prior, mask, theta_map, hess, gtol,
                                      args.maxiter, n_panel=2 * args.panel_nodes,
                                      init_names=("laplace",))
            quad_drift = float(np.sum((best["mu"] - fine["mu"]) ** 2))
            # All verification gates enforce BEFORE any gold comparison.
            enforce_vi_gates((ds, prior), sols, agree, quad_drift, min_s_final)
            gold_mean = np.asarray(g["gold_mean"], dtype=np.float64)
            vi_err = float(np.sum((best["mu"] - gold_mean) ** 2))
            rows.append({
                "kind": "vi_baseline",
                "target": "rr_extrapolation",
                "dataset": ds,
                "prior": prior,
                "dim": d,
                "n_train": int(x.shape[0]),
                "gold_id": g["gold_id"],
                "gl_panel_nodes": args.panel_nodes,
                "vi_err_sq_gold": vi_err,
                "gold_spread": g["gold_spread"],
                "zs_ula_eta1_median": zs_med,
                "zs_ula_eta1_reps": zs_reps,
                "vi_over_zs_ratio": vi_err / zs_med,
                "map_err_sq_gold": float(np.sum((theta_map - gold_mean) ** 2)),
                "trace_sigma_vi": float(np.sum(best["L"] * best["L"])),
                "neg_elbo": best["neg_elbo"],
                "grad_inf_norm": best["grad_inf"],
                "lbfgs_iters": {k: v["n_iter"] for k, v in sols.items()},
                "lbfgs_status": {k: [v["success"], v["message"]] for k, v in sols.items()},
                "init_agreement_sq": agree,
                "quad_drift_sq": quad_drift,
                "min_s_seen": min_s_seen,
                "min_s_final": min_s_final,
                "gates": VI_GATES,
                "provenance": prov,
                "mu_vi": [float(v) for v in best["mu"]],
                "git_hash": git_hash(),
            })
            print(f"VI  {ds:9s} prior {prior:<4g} err^2 {vi_err:12.6g} "
                  f"(MAP {rows[-1]['map_err_sq_gold']:.6g}, zs {zs_med:.4g}) "
                  f"agree {agree:.2e} quad_drift {quad_drift:.2e} "
                  f"min_s {min_s_final:.3g} gnorm {best['grad_inf']:.1e} "
                  f"iters {rows[-1]['lbfgs_iters']}")

            if not args.skip_ep:
                ep, converged = fit_ep(x, y, prior, mask, theta_map, max_sweeps=max_sweeps,
                                       n_panel=args.panel_nodes)
                ep_all_converged &= converged
                if converged:
                    ep_fine, conv_fine = fit_ep(x, y, prior, mask, theta_map,
                                                max_sweeps=max_sweeps,
                                                n_panel=2 * args.panel_nodes)
                    ep_quad_drift = (float(np.sum((ep["mu"] - ep_fine["mu"]) ** 2))
                                     if conv_fine else float("nan"))
                    ep_err = float(np.sum((ep["mu"] - gold_mean) ** 2))
                    rows.append({
                        "kind": "ep_baseline",
                        "target": "rr_extrapolation",
                        "dataset": ds,
                        "prior": prior,
                        "dim": d,
                        "n_train": int(x.shape[0]),
                        "gold_id": g["gold_id"],
                        "gl_panel_nodes": args.panel_nodes,
                        "quad_drift_sq": ep_quad_drift,
                        "ep_err_sq_gold": ep_err,
                        "gold_spread": g["gold_spread"],
                        "zs_ula_eta1_median": zs_med,
                        "ep_over_zs_ratio": ep_err / zs_med,
                        "trace_sigma_ep": ep["trace_sigma"],
                        "ep_sweeps": ep["sweeps"],
                        "ep_damping": ep["damping"],
                        "ep_delta_final": ep["delta"],
                        "ep_cavity_skips": ep["n_skipped"],
                        "ep_min_tau": ep["min_tau"],
                        "ep_mu_step_final": ep["mu_step"],
                        "converged": True,
                        "provenance": prov,
                        "mu_ep": [float(v) for v in ep["mu"]],
                        "git_hash": git_hash(),
                    })
                    print(f"EP  {ds:9s} prior {prior:<4g} err^2 {ep_err:12.6g} "
                          f"sweeps {ep['sweeps']} damping {ep['damping']} "
                          f"min_tau {ep['min_tau']:.2e} quad_drift {ep_quad_drift:.2e}")
                else:
                    rows.append({"kind": "ep_baseline", "dataset": ds, "prior": prior,
                                 "converged": False, "ep_sweeps": ep["sweeps"],
                                 "ep_damping": ep.get("damping"), "ep_delta_final": ep.get("delta"),
                                 "ep_cavity_skips": ep.get("n_skipped"),
                                 "ep_min_tau": ep.get("min_tau"),
                                 "ep_mu_step_final": ep.get("mu_step"),
                                 "ep_fail_reason": ep.get("reason"),
                                 "provenance": prov, "git_hash": git_hash()})
                    print(f"EP  {ds:9s} prior {prior:<4g} DID NOT CONVERGE ({ep.get('reason')})")

    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}")

    print("\nVI/EP columns for paper/tables/gold_endtask.tex ('{:.3g}' style, table row order):")
    vi_idx = {(r["dataset"], r["prior"]): r for r in rows if r["kind"] == "vi_baseline"}
    ep_idx = {(r["dataset"], r["prior"]): r for r in rows if r["kind"] == "ep_baseline" and r.get("converged")}
    for ds in datasets:
        for prior in PRIORS:
            v = vi_idx[(ds, prior)]
            e = ep_idx.get((ds, prior))
            ep_cell = f"{e['ep_err_sq_gold']:.3g}" if e else "---"
            print(f"{DATASET_LABEL[ds]} & {prior}  ->  VI: {v['vi_err_sq_gold']:.3g}   EP: {ep_cell}")
    if not args.skip_ep and not ep_all_converged:
        print("NOTE: EP failed to converge on at least one cell -> per pre-commit, drop the EP column.")


if __name__ == "__main__":
    main()
