import argparse
import math

import numpy as np


def make_random_correlation(rng, k, rank=None):
    rank = k if rank is None else rank
    a = rng.normal(size=(k, rank))
    c = a @ a.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def zero_sum_covariance(k):
    return (k / (k - 1.0)) * (np.eye(k) - np.ones((k, k)) / k)


def relative_error(observed, expected):
    denom = max(1.0, abs(float(expected)))
    return abs(float(observed) - float(expected)) / denom


def check_quadratic_ensemble_variance(rng, trials):
    k = 7
    c = make_random_correlation(rng, k, rank=4)
    x = rng.multivariate_normal(np.zeros(k), c, size=trials)
    observed = np.var(x.mean(axis=1), ddof=0)
    expected = float(np.ones(k) @ c @ np.ones(k) / (k * k))
    err = relative_error(observed, expected)
    return {
        "name": "quadratic ensemble variance",
        "observed": observed,
        "expected": expected,
        "rel_error": err,
        "passed": err < 0.035,
    }


def check_weighted_estimator_identity(rng, trials):
    k = 8
    dim = 3
    steps = 9
    eta = 0.06
    c = make_random_correlation(rng, k, rank=5)
    h = np.array([0.8, 1.4, 2.1])
    a = rng.normal(size=dim)
    weights = rng.uniform(0.05, 1.0, size=steps + 1)
    evals, evecs = np.linalg.eigh(c)
    evals = np.clip(evals, 0.0, None)
    c_half = evecs @ np.diag(np.sqrt(evals))

    x = np.zeros((trials, k, dim))
    estimates = weights[0] * np.einsum("td,d->t", x.mean(axis=1), a)
    for t in range(steps):
        z = rng.normal(size=(trials, k, dim))
        xi = np.einsum("tjd,kj->tkd", z, c_half)
        x = x * (1.0 - eta * h) + math.sqrt(2.0 * eta) * xi
        estimates = estimates + weights[t + 1] * np.einsum("td,d->t", x.mean(axis=1), a)

    observed = np.var(estimates, ddof=0)
    factor = float(np.ones(k) @ c @ np.ones(k) / (k * k))
    expected = 0.0
    for s in range(steps):
        g = np.zeros(dim)
        for t in range(s + 1, steps + 1):
            g += weights[t] * ((1.0 - eta * h) ** (t - 1 - s))
        expected += 2.0 * eta * float((a * g) @ (a * g))
    expected *= factor
    err = relative_error(observed, expected)
    return {
        "name": "time-weighted estimator identity",
        "observed": observed,
        "expected": expected,
        "rel_error": err,
        "passed": err < 0.035,
    }


def check_zero_sum_construction(rng, trials):
    k = 9
    z = rng.normal(size=(trials, k))
    z = z - z.mean(axis=1, keepdims=True)
    z = z * math.sqrt(k / (k - 1.0))
    observed = np.cov(z, rowvar=False, bias=True)
    expected = zero_sum_covariance(k)
    cov_max_abs = float(np.max(np.abs(observed - expected)))
    sum_max_abs = float(np.max(np.abs(z.sum(axis=1))))
    return {
        "name": "zero-sum Gaussian construction",
        "cov_max_abs": cov_max_abs,
        "sum_max_abs": sum_max_abs,
        "passed": cov_max_abs < 0.018 and sum_max_abs < 1e-12,
    }


def cosine_grad(x, beta, omega):
    return x + (beta / omega) * np.sin(omega * x)


def check_curvature_spread_pathwise_bound(rng):
    reps = 256
    k = 8
    dim = 3
    steps = 120
    eta = 0.08
    beta = 0.35
    omega = 1.7
    m = 1.0 - beta
    rho = beta * omega
    q = max(abs(1.0 - eta * m), abs(1.0 - eta * (1.0 + beta)))

    x0 = rng.normal(scale=0.2, size=(reps, 1, dim))
    x = np.repeat(x0, k, axis=1)
    y = x0[:, 0, :].copy()
    bound = np.zeros(reps)

    for _ in range(steps):
        center = x.mean(axis=1)
        spread = np.mean(np.sum((x - center[:, None, :]) ** 2, axis=2), axis=1)
        z = rng.normal(size=(reps, k, dim))
        z = z - z.mean(axis=1, keepdims=True)
        z = z * math.sqrt(k / (k - 1.0))

        mean_remainder = np.mean(cosine_grad(x, beta, omega), axis=1) - cosine_grad(center, beta, omega)
        # This verifies the local Taylor ingredient used by the recursive path bound.
        local_ok = np.linalg.norm(mean_remainder, axis=1) <= 0.5 * rho * spread + 5e-13
        if not bool(np.all(local_ok)):
            return {
                "name": "curvature-spread pathwise bound",
                "max_ratio": float("inf"),
                "passed": False,
            }

        bound = q * bound + 0.5 * eta * rho * spread
        x = x - eta * cosine_grad(x, beta, omega) + math.sqrt(2.0 * eta) * z
        y = y - eta * cosine_grad(y, beta, omega)

    residual = np.linalg.norm(x.mean(axis=1) - y, axis=1)
    ratio = residual / np.maximum(bound, 1e-14)
    return {
        "name": "curvature-spread pathwise bound",
        "residual_max": float(np.max(residual)),
        "bound_min": float(np.min(bound)),
        "max_ratio": float(np.max(ratio)),
        "passed": bool(np.all(residual <= bound + 2e-10)),
    }


def check_perturbative_variance_scaling(rng, trials):
    k = 8
    steps = 80
    eta = 0.05
    omega = 1.3
    epsilons = np.array([0.025, 0.05, 0.1, 0.2])
    variances = []
    base_noise = []

    for _ in range(steps):
        z = rng.normal(size=(trials, k))
        z = z - z.mean(axis=1, keepdims=True)
        z = z * math.sqrt(k / (k - 1.0))
        base_noise.append(z)

    for eps in epsilons:
        x = np.zeros((trials, k))
        y = np.zeros(trials)
        for z in base_noise:
            x = x - eta * (x + (eps / omega) * np.sin(omega * x)) + math.sqrt(2.0 * eta) * z
            y = y - eta * (y + (eps / omega) * np.sin(omega * y))
        variances.append(float(np.var(x.mean(axis=1) - y)))

    variances = np.array(variances)
    slope = float(np.polyfit(np.log(epsilons), np.log(variances), 1)[0])
    normalized = variances / (epsilons * epsilons)
    ratio = float(np.max(normalized) / np.min(normalized))
    return {
        "name": "sinusoidal perturbation variance scaling",
        "epsilons": epsilons,
        "variances": variances,
        "loglog_slope": slope,
        "normalized_ratio": ratio,
        "passed": 1.75 <= slope <= 2.25 and ratio < 2.25,
    }


def format_value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=4, separator=", ")
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Quick NumPy sanity checks for zero-sum Langevin theory identities.")
    parser.add_argument("--trials", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260502)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    checks = [
        check_quadratic_ensemble_variance(rng, args.trials),
        check_weighted_estimator_identity(rng, args.trials),
        check_zero_sum_construction(rng, args.trials),
        check_curvature_spread_pathwise_bound(rng),
        check_perturbative_variance_scaling(rng, max(args.trials // 4, 20_000)),
    ]

    all_passed = True
    for result in checks:
        passed = bool(result.pop("passed"))
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        details = ", ".join(f"{k}={format_value(v)}" for k, v in result.items() if k != "name")
        print(f"{status} {result['name']}: {details}")

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
