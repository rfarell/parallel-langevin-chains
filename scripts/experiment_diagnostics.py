import argparse
import math
from pathlib import Path

import numpy as np


DEFAULT_SEED = 20260502
DEFAULT_SAMPLES = 100_000


def ensure_psd_sqrt(cov):
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 0.0, None)
    return evecs @ np.diag(np.sqrt(evals))


def equicorrelation(k, rho):
    return (1.0 - rho) * np.eye(k) + rho * np.ones((k, k))


def zero_sum_noise(rng, shape):
    z = rng.normal(size=shape)
    z = z - z.mean(axis=1, keepdims=True)
    return z * math.sqrt(shape[1] / (shape[1] - 1.0))


def coupled_noise(rng, shape, method):
    if method == "iid":
        return rng.normal(size=shape)
    if method == "zero-sum":
        return zero_sum_noise(rng, shape)
    if method == "common":
        z = rng.normal(size=(shape[0], 1, shape[2]))
        return np.repeat(z, shape[1], axis=1)
    if method == "antithetic":
        z = rng.normal(size=(shape[0], shape[1] // 2, shape[2]))
        return np.concatenate([z, -z], axis=1)
    raise ValueError(method)


def normal_var_ci(var_hat, n):
    half = 1.96 * math.sqrt(2.0 / max(n - 1, 1)) * var_hat
    return max(0.0, var_hat - half), var_hat + half


def format_num(x):
    x = float(x)
    if x == 0.0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e3:
        return f"{x:.2e}"
    return f"{x:.4f}"


def format_ci(x, lo, hi):
    return f"{format_num(x)} [{format_num(lo)}, {format_num(hi)}]"


def write_table(path, comment, headers, rows):
    align = "l" + "r" * (len(headers) - 1)
    lines = [
        f"% {comment}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(headers) + r" \\",
        "\\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def diagnostic_correlation_sweep(rng, out_dir, samples):
    k = 8
    rhos = [-1.0 / (k - 1.0), -0.10, 0.0, 0.20, 0.50, 0.80, 1.0]
    rows = []
    for rho in rhos:
        cov = equicorrelation(k, rho)
        z = rng.normal(size=(samples, k)) @ ensure_psd_sqrt(cov).T
        means = z.mean(axis=1)
        var_hat = float(np.var(means, ddof=1))
        theory = float(np.ones(k) @ cov @ np.ones(k) / (k * k))
        lo, hi = normal_var_ci(var_hat, samples)
        rows.append([format_num(rho), format_num(theory), format_ci(var_hat, lo, hi), format_num(var_hat - theory)])
    write_table(
        out_dir / "diagnostic_correlation_sweep.tex",
        f"Equicorrelation sweep; k={k}, samples={samples}, seed stream from {DEFAULT_SEED}; MC intervals are approximate 95% CIs.",
        [r"$\rho$", "exact Var(mean)", "MC Var(mean)", "MC-exact"],
        rows,
    )


def weighted_variance_theory(k, cov, eta, h, a, weights):
    steps = len(weights) - 1
    factor = float(np.ones(k) @ cov @ np.ones(k) / (k * k))
    total = 0.0
    decay = 1.0 - eta * h
    for s in range(steps):
        g = np.zeros_like(a)
        for t in range(s + 1, steps + 1):
            g += weights[t] * decay ** (t - 1 - s)
        total += 2.0 * eta * float((a * g) @ (a * g))
    return factor * total


def diagnostic_weighted_timeavg(rng, out_dir, samples):
    k = 8
    dim = 3
    steps = 10
    eta = 0.055
    h = np.array([0.7, 1.3, 2.2])
    a = np.array([0.8, -0.4, 1.1])
    cov = equicorrelation(k, 0.35)
    cov_sqrt = ensure_psd_sqrt(cov)
    weight_sets = {
        "uniform": np.ones(steps + 1) / (steps + 1),
        "front": np.linspace(2.0, 0.4, steps + 1),
        "back": np.linspace(0.4, 2.0, steps + 1),
    }
    rows = []
    for name, raw_weights in weight_sets.items():
        weights = raw_weights / raw_weights.sum()
        x = np.zeros((samples, k, dim))
        estimate = weights[0] * (x.mean(axis=1) @ a)
        for t in range(steps):
            z = rng.normal(size=(samples, k, dim))
            xi = np.einsum("nkd,jk->njd", z, cov_sqrt)
            x = x * (1.0 - eta * h) + math.sqrt(2.0 * eta) * xi
            estimate += weights[t + 1] * (x.mean(axis=1) @ a)
        var_hat = float(np.var(estimate, ddof=1))
        theory = weighted_variance_theory(k, cov, eta, h, a, weights)
        lo, hi = normal_var_ci(var_hat, samples)
        rows.append([name, format_num(theory), format_ci(var_hat, lo, hi), format_num(var_hat / theory)])
    write_table(
        out_dir / "diagnostic_weighted_timeavg.tex",
        f"Weighted time-average quadratic variance identity; k={k}, dim={dim}, steps={steps}, eta={eta}, samples={samples}, seed stream from {DEFAULT_SEED}.",
        ["weights", "exact variance", "MC variance", "MC/exact"],
        rows,
    )


def diagnostic_marginal_invariance(rng, out_dir, samples):
    k = 8
    dim = 5
    steps = 80
    eta = 0.04
    h = np.array([0.6, 0.9, 1.2, 1.7, 2.4])
    probe = np.array([0.4, -0.7, 0.1, 0.5, -0.2])
    probe = probe / np.linalg.norm(probe)
    decay = 1.0 - eta * h
    exact_second = float(np.mean(2.0 * eta * (1.0 - decay ** (2 * steps)) / (1.0 - decay * decay)))
    rows = []
    reference_probe = None
    for method in ["iid", "zero-sum", "antithetic", "common"]:
        x = np.zeros((samples, k, dim))
        for _ in range(steps):
            z = coupled_noise(rng, (samples, k, dim), method)
            x = x * (1.0 - eta * h) + math.sqrt(2.0 * eta) * z
        obs = np.mean(x[:, 0, :] * x[:, 0, :], axis=1)
        probe_obs = x[:, 0, :] @ probe
        if reference_probe is None:
            reference_probe = np.sort(probe_obs)
        probe_sorted = np.sort(probe_obs)
        grid = np.sort(np.concatenate([reference_probe, probe_sorted]))
        ref_cdf = np.searchsorted(reference_probe, grid, side="right") / samples
        probe_cdf = np.searchsorted(probe_sorted, grid, side="right") / samples
        ks = float(np.max(np.abs(ref_cdf - probe_cdf)))
        mean_hat = float(np.mean(obs))
        se = float(np.std(obs, ddof=1) / math.sqrt(samples))
        rows.append(
            [
                method,
                format_num(exact_second),
                format_ci(mean_hat, mean_hat - 1.96 * se, mean_hat + 1.96 * se),
                format_num(mean_hat / exact_second),
                format_num(ks),
            ]
        )
    write_table(
        out_dir / "diagnostic_marginal_invariance.tex",
        f"Single-chain marginal invariance; k={k}, dim={dim}, steps={steps}, eta={eta}, samples={samples}, seed stream from {DEFAULT_SEED}.",
        ["coupling", r"exact $E\|X_1\|^2/d$", "MC estimate", "MC/exact", "KS vs iid"],
        rows,
    )


def cosine_grad(x, beta, omega):
    return x + (beta / omega) * np.sin(omega * x)


def diagnostic_nonlinear(rng, out_dir, samples):
    k = 8
    dim = 3
    steps = 90
    eta = 0.05
    reps = min(samples, 50_000)
    rows = []
    for beta, omega in [(0.15, 1.0), (0.35, 1.7)]:
        rho_lip = beta * omega
        q = max(abs(1.0 - eta * (1.0 - beta)), abs(1.0 - eta * (1.0 + beta)))
        x0 = rng.normal(scale=0.15, size=(reps, 1, dim))
        x = np.repeat(x0, k, axis=1)
        y = x0[:, 0, :].copy()
        bound = np.zeros(reps)
        local_max = 0.0
        for _ in range(steps):
            center = x.mean(axis=1)
            spread = np.mean(np.sum((x - center[:, None, :]) ** 2, axis=2), axis=1)
            remainder = np.mean(cosine_grad(x, beta, omega), axis=1) - cosine_grad(center, beta, omega)
            local = np.linalg.norm(remainder, axis=1) / np.maximum(0.5 * rho_lip * spread, 1e-14)
            local_max = max(local_max, float(np.max(local)))
            bound = q * bound + 0.5 * eta * rho_lip * spread
            z = zero_sum_noise(rng, (reps, k, dim))
            x = x - eta * cosine_grad(x, beta, omega) + math.sqrt(2.0 * eta) * z
            y = y - eta * cosine_grad(y, beta, omega)
        residual = np.linalg.norm(x.mean(axis=1) - y, axis=1)
        path_ratio = residual / np.maximum(bound, 1e-14)
        rows.append([f"{beta:.2f}", f"{omega:.1f}", format_num(local_max), format_num(np.max(path_ratio)), format_num(np.mean(residual))])

    epsilons = np.array([0.025, 0.05, 0.10, 0.20])
    variances = []
    base_noise = [zero_sum_noise(rng, (reps, k, 1))[:, :, 0] for _ in range(steps)]
    for eps in epsilons:
        x = np.zeros((reps, k))
        y = np.zeros(reps)
        for z in base_noise:
            x = x - eta * (x + eps * np.sin(x)) + math.sqrt(2.0 * eta) * z
            y = y - eta * (y + eps * np.sin(y))
        variances.append(float(np.var(x.mean(axis=1) - y, ddof=1)))
    slope = float(np.polyfit(np.log(epsilons), np.log(variances), 1)[0])
    rows.append(["pert.", "--", "--", format_num(slope), "--"])
    write_table(
        out_dir / "diagnostic_nonlinear.tex",
        f"Nonlinear bounded-derivative diagnostics; k={k}, steps={steps}, eta={eta}, samples={reps}, seed stream from {DEFAULT_SEED}.",
        [r"$\beta$", r"$\omega$", "local bound ratio", "path ratio/slope", "mean resid."],
        rows,
    )


def diagnostic_quadratic_observable(rng, out_dir, samples):
    k = 8
    dim = 64
    alpha = 1.0
    betas = [0.0, 0.5, 1.0, 2.0]
    rows = []
    z = zero_sum_noise(rng, (samples, k, dim))
    probe = rng.normal(size=dim)
    probe /= np.linalg.norm(probe)
    linear = np.einsum("nkd,d->nk", z, probe)
    quadratic = (np.sum(z * z, axis=2) - dim) / math.sqrt(2.0 * dim)
    for beta in betas:
        obs = alpha * linear + beta * quadratic
        mean_obs = obs.mean(axis=1)
        chain_obs = obs[:, 0]
        ratio = float(np.var(mean_obs, ddof=1) / np.var(chain_obs, ddof=1))
        theory = (beta * beta / (k - 1.0)) / (alpha * alpha + beta * beta)
        lo, hi = normal_var_ci(ratio, samples)
        rows.append([format_num(beta), format_num(theory), format_ci(ratio, lo, hi), format_num(ratio / theory) if theory > 0 else "--"])
    write_table(
        out_dir / "diagnostic_quadratic_observable.tex",
        f"Quadratic observable zero-sum failure mode; k={k}, dim={dim}, alpha={alpha}, samples={samples}, seed stream from {DEFAULT_SEED}.",
        [r"quadratic $\beta$", "exact ratio", "MC ratio", "MC/exact"],
        rows,
    )


def diagonal_spectrum(dim, condition):
    if condition == 1:
        return np.ones(dim)
    return np.exp(math.log(condition) * np.linspace(0.0, 1.0, dim))


def make_quadratic_components(rng, dim, n_data, condition, hessian_jitter, linear_noise):
    full_h = diagonal_spectrum(dim, condition)
    if hessian_jitter == 0.0:
        comp_h = np.repeat((full_h / n_data)[None, :], n_data, axis=0)
    else:
        eps = rng.normal(size=(n_data, dim))
        eps = eps - eps.mean(axis=0, keepdims=True)
        comp_h = (full_h[None, :] / n_data) * np.maximum(0.05, 1.0 + hessian_jitter * eps)
        full_h = comp_h.sum(axis=0)
    comp_c = linear_noise * rng.normal(size=(n_data, dim)) / (n_data * math.sqrt(dim))
    full_c = comp_c.sum(axis=0)
    return comp_h, comp_c, full_h, full_c


def iid_minibatches(rng, reps, chains, n_data, batch_size):
    return rng.integers(0, n_data, size=(reps, chains, batch_size))


def balanced_minibatches(rng, reps, chains, n_data, batch_size):
    keys = rng.random(size=(reps, n_data))
    perm = np.argsort(keys, axis=1)
    return perm.reshape(reps, chains, batch_size)


def diagnostic_sgld(rng, out_dir, samples):
    reps = min(samples // 50, 2_000)
    k = 8
    dim = 8
    batch_size = 8
    n_data = k * batch_size
    steps = 70
    eta = 0.018
    condition = 3
    linear_noise = 1.0
    probe = rng.normal(size=dim)
    probe /= np.linalg.norm(probe)
    rows = []
    for hessian_label, hessian_jitter in [("common", 0.0), ("hetero", 0.30)]:
        comp_h, comp_c, full_h, full_c = make_quadratic_components(
            rng, dim, n_data, condition, hessian_jitter, linear_noise
        )
        target = full_c / full_h
        for balanced in [False, True]:
            for zero_sum in [False, True]:
                x = np.zeros((reps, k, dim))
                rem_acc = 0.0
                rem_count = 0
                for step in range(steps):
                    if balanced:
                        idx = balanced_minibatches(rng, reps, k, n_data, batch_size)
                    else:
                        idx = iid_minibatches(rng, reps, k, n_data, batch_size)
                    batch_h = (n_data / batch_size) * comp_h[idx].sum(axis=2)
                    batch_c = (n_data / batch_size) * comp_c[idx].sum(axis=2)
                    grad = batch_h * x - batch_c
                    if step == steps - 1 or step % 24 == 0:
                        center = x.mean(axis=1)
                        rem = grad.mean(axis=1) - (full_h[None, :] * center - full_c[None, :])
                        rem_acc += float(np.sqrt(np.mean(np.sum(rem * rem, axis=1))))
                        rem_count += 1
                    z = rng.normal(size=x.shape)
                    if zero_sum:
                        z = z - z.mean(axis=1, keepdims=True)
                        z = z * math.sqrt(k / (k - 1.0))
                    x = x - eta * grad + math.sqrt(2.0 * eta) * z
                ens = x.mean(axis=1)
                chain0 = x[:, 0, :]
                lin_ens = (ens - target[None, :]) @ probe
                lin_chain = (chain0 - target[None, :]) @ probe
                ratio = float(np.var(lin_ens, ddof=1) / np.var(lin_chain, ddof=1))
                lo, hi = normal_var_ci(ratio, reps)
                reference = 0.0 if zero_sum else 1.0 / k
                rows.append(
                    [
                        hessian_label,
                        "balanced" if balanced else "iid",
                        "zero-sum" if zero_sum else "iid",
                        format_num(reference),
                        format_ci(ratio, lo, hi),
                        format_num(rem_acc / rem_count),
                    ]
                )
    write_table(
        out_dir / "diagnostic_sgld.tex",
        f"Finite-sum quadratic SGLD 2x2 diagnostic; k={k}, dim={dim}, n={n_data}, batch={batch_size}, steps={steps}, eta={eta}, samples={reps}, seed stream from {DEFAULT_SEED}. Reference ratio is the innovation-only linear theory.",
        ["Hessian", "minibatch", "innovation", "ref. ratio", "terminal ratio", "rem. norm"],
        rows,
    )


def main():
    parser = argparse.ArgumentParser(description="Focused deterministic NumPy diagnostics for zero-sum Langevin experiments.")
    parser.add_argument("--out-dir", type=Path, default=Path("paper/tables"))
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    diagnostic_correlation_sweep(rng, args.out_dir, args.samples)
    diagnostic_weighted_timeavg(rng, args.out_dir, args.samples)
    diagnostic_marginal_invariance(rng, args.out_dir, args.samples)
    diagnostic_nonlinear(rng, args.out_dir, args.samples)
    diagnostic_quadratic_observable(rng, args.out_dir, args.samples)
    diagnostic_sgld(rng, args.out_dir, args.samples)
    print(f"Wrote diagnostic tables to {args.out_dir}")


if __name__ == "__main__":
    main()
