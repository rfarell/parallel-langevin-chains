"""Measure the actual multi-GPU cost of across-chain zero-sum centering.

Shards K ULA chains of the frozen spambase/adult logistic loops across local
A100s (PCIe, no NVLink). Per step each GPU computes gradients for its chain
shard and draws its innovations; zero_sum additionally all-reduces the
per-GPU innovation sums to form the global innovation mean. All-reduce uses
cupy.cuda.nccl when loadable (on this system: module load cuda/12.2
nccl/2.18.5); otherwise falls back to host staging through pinned memory
(async D2H per stream, host-side sum, async H2D broadcast). Timing only.
"""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import cupy as cp
import numpy as np

try:
    from cupy.cuda import nccl
    NCCL_AVAILABLE = bool(getattr(nccl, "available", False))
except Exception:
    nccl = None
    NCCL_AVAILABLE = False


DATASET_DEFAULTS = {"spambase": {"reps": 192}, "adult": {"reps": 64}}
SEED_BASE = {"spambase": 660000, "adult": 670000}


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def sigmoid(z):
    return 1.0 / (1.0 + cp.exp(-cp.clip(z, -30.0, 30.0)))


def add_intercept(x_np):
    ones = np.ones((x_np.shape[0], 1), dtype=np.float32)
    return np.concatenate([ones, x_np.astype(np.float32)], axis=1)


def load_dataset(data_dir, name):
    data = np.load(data_dir / f"{name}.npz", allow_pickle=True)
    return add_intercept(data["X_train"]), data["y_train"].astype(np.float32)


def logistic_grad(x_state, data_x, data_y, prior, chunk_rows, prior_mask):
    flat = x_state.reshape((-1, x_state.shape[-1]))
    out = cp.empty_like(flat)
    n = data_x.shape[0]
    for start in range(0, flat.shape[0], chunk_rows):
        stop = min(start + chunk_rows, flat.shape[0])
        part = flat[start:stop]
        probs = sigmoid(part @ data_x.T)
        grad = (probs - data_y[None, :]) @ data_x / n
        grad = grad + prior * part * prior_mask[None, :]
        out[start:stop] = grad
    return out.reshape(x_state.shape)


def shard_chains(k, n_gpus):
    base, rem = divmod(k, n_gpus)
    return [base + (1 if i < rem else 0) for i in range(n_gpus)]


class Worker:
    def __init__(self, dev, x_np, y_np, reps, k_local, dim, seed):
        self.dev = dev
        with cp.cuda.Device(dev):
            cp.random.seed(seed)
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.data_x = cp.asarray(x_np, dtype=cp.float32)
            self.data_y = cp.asarray(y_np, dtype=cp.float32)
            self.prior_mask = cp.ones(dim, dtype=cp.float32)
            self.prior_mask[0] = 0.0
            self.x = cp.zeros((reps, k_local, dim), dtype=cp.float32)
            self.grad = None
            self.z = None
            self.sum_buf = cp.zeros((reps, dim), dtype=cp.float32)

    def reset(self, reps, k_local, dim, seed):
        with cp.cuda.Device(self.dev):
            cp.random.seed(seed)
            self.x = cp.zeros((reps, k_local, dim), dtype=cp.float32)


def sync_all(workers):
    for w in workers:
        with cp.cuda.Device(w.dev):
            w.stream.synchronize()


class NcclAllReduce:
    def __init__(self, workers, reps, dim):
        self.count = reps * dim
        self.comms = nccl.NcclCommunicator.initAll([w.dev for w in workers])

    def __call__(self, workers):
        nccl.groupStart()
        for w, comm in zip(workers, self.comms):
            with cp.cuda.Device(w.dev):
                comm.allReduce(w.sum_buf.data.ptr, w.sum_buf.data.ptr, self.count,
                               nccl.NCCL_FLOAT32, nccl.NCCL_SUM, w.stream.ptr)
        nccl.groupEnd()


class HostStagingAllReduce:
    """Fallback: async D2H into pinned buffers, host-side sum, async H2D."""

    def __init__(self, workers, reps, dim):
        nbytes = reps * dim * 4
        self.shape = (reps, dim)
        self.host = []
        for _ in workers:
            mem = cp.cuda.alloc_pinned_memory(nbytes)
            self.host.append(np.frombuffer(mem, dtype=np.float32, count=reps * dim).reshape(self.shape))
        mem = cp.cuda.alloc_pinned_memory(nbytes)
        self.total = np.frombuffer(mem, dtype=np.float32, count=reps * dim).reshape(self.shape)

    def __call__(self, workers):
        for w, h in zip(workers, self.host):
            with cp.cuda.Device(w.dev):
                w.sum_buf.get(stream=w.stream, out=h)
        sync_all(workers)
        np.copyto(self.total, self.host[0])
        for h in self.host[1:]:
            self.total += h
        for w in workers:
            with cp.cuda.Device(w.dev):
                w.sum_buf.set(self.total, stream=w.stream)


def step_iid(workers, eta, sqrt_2eta, prior, chunk_rows, k, all_reduce):
    for w in workers:
        with cp.cuda.Device(w.dev), w.stream:
            grad = logistic_grad(w.x, w.data_x, w.data_y, prior, chunk_rows, w.prior_mask)
            z = cp.random.standard_normal(w.x.shape, dtype=cp.float32)
            w.x = w.x - eta * grad + sqrt_2eta * z


def step_zero_sum(workers, eta, sqrt_2eta, prior, chunk_rows, k, all_reduce):
    for w in workers:
        with cp.cuda.Device(w.dev), w.stream:
            w.grad = logistic_grad(w.x, w.data_x, w.data_y, prior, chunk_rows, w.prior_mask)
            w.z = cp.random.standard_normal(w.x.shape, dtype=cp.float32)
            w.sum_buf = w.z.sum(axis=1)
    all_reduce(workers)
    scale = math.sqrt(k / (k - 1))
    for w in workers:
        with cp.cuda.Device(w.dev), w.stream:
            mean = w.sum_buf * (1.0 / k)
            w.x = w.x - eta * w.grad + sqrt_2eta * ((w.z - mean[:, None, :]) * scale)


STEP_FNS = {"iid": step_iid, "zero_sum": step_zero_sum}


def time_method(method, workers, config, all_reduce):
    step_fn = STEP_FNS[method]
    args = (workers, config["eta"], math.sqrt(2.0 * config["eta"]), config["prior"],
            config["chunk_rows"], config["chains"], all_reduce)
    for w, k_local in zip(workers, config["chains_per_gpu"]):
        w.reset(config["reps"], k_local, config["dim"], config["seed"] + 13 * w.dev + (7 if method == "zero_sum" else 0))
    for _ in range(config["warmup"]):
        step_fn(*args)
    sync_all(workers)
    elapsed = []
    for _ in range(config["timing_reps"]):
        start = time.perf_counter()
        for _ in range(config["steps"]):
            step_fn(*args)
        sync_all(workers)
        elapsed.append(time.perf_counter() - start)
    return elapsed


def run_config(dataset, k, args, x_np, y_np, out_file):
    n_gpus = len(args.devices)
    chains_per_gpu = shard_chains(k, n_gpus)
    reps = args.reps or max(24, DATASET_DEFAULTS[dataset]["reps"] // max(1, k // 8))
    dim = x_np.shape[1]
    eta = args.eta if args.eta is not None else min(0.03 / (args.prior + 0.25), 0.08)
    seed = SEED_BASE[dataset] + 1000 * k
    config = {
        "dataset": dataset, "chains": k, "chains_per_gpu": chains_per_gpu, "reps": reps,
        "dim": dim, "prior": args.prior, "eta": eta, "steps": args.steps, "warmup": args.warmup,
        "timing_reps": args.timing_reps, "chunk_rows": args.chunk_rows, "seed": seed,
    }
    workers = [Worker(dev, x_np, y_np, reps, k_local, dim, seed + 13 * dev)
               for dev, k_local in zip(args.devices, chains_per_gpu)]
    nccl_used = NCCL_AVAILABLE and not args.no_nccl
    if nccl_used:
        all_reduce = NcclAllReduce(workers, reps, dim)
    else:
        all_reduce = HostStagingAllReduce(workers, reps, dim)
    base = {
        **config,
        "target": "distributed_overhead",
        "n_gpus": n_gpus,
        "devices": list(args.devices),
        "nccl_used": nccl_used,
        "reduce_impl": "nccl" if nccl_used else "host_staging_pinned",
        "topology": args.topology,
        "git_hash": git_hash(),
    }
    iid_elapsed = time_method("iid", workers, config, all_reduce)
    zs_elapsed = time_method("zero_sum", workers, config, all_reduce)
    for method, elapsed_list in [("iid", iid_elapsed), ("zero_sum", zs_elapsed)]:
        for rep_idx, elapsed in enumerate(elapsed_list):
            row = {
                **base,
                "method": method,
                "coupling": method,
                "timing_rep": rep_idx,
                "elapsed_sec": elapsed,
                "sec_per_step": elapsed / config["steps"],
                "sec_per_ensemble_step": elapsed / (config["steps"] * reps),
                "overhead_ratio_vs_iid": (elapsed / iid_elapsed[rep_idx]) if method == "zero_sum" else None,
            }
            out_file.write(json.dumps(row, sort_keys=True) + "\n")
            out_file.flush()
            print(json.dumps(row, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["spambase", "adult"], default=["spambase", "adult"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k-values", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--devices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timing-reps", type=int, default=5)
    parser.add_argument("--prior", type=float, default=0.3)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--no-nccl", action="store_true")
    parser.add_argument("--topology", default="3x A100-PCIE-40GB, PCIe SYS links (no NVLink); GPU0 NUMA0, GPU1-2 NUMA1")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.datasets = ["spambase"]
        args.k_values = [8]
        args.steps = 50
        args.warmup = 10
        args.timing_reps = 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"nccl available: {NCCL_AVAILABLE}", flush=True)
    with args.out.open("a", encoding="utf-8") as f:
        for dataset in args.datasets:
            x_np, y_np = load_dataset(args.data_dir, dataset)
            for k in args.k_values:
                run_config(dataset, k, args, x_np, y_np, f)


if __name__ == "__main__":
    main()
