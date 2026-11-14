import argparse
import gzip
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np


MNIST_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

DATASETS = {
    "mnist79": {
        "url": MNIST_BASE,
        "citation": "LeCun, Cortes, and Burges, THE MNIST DATABASE of handwritten digits (mirror: ossci-datasets.s3.amazonaws.com/mnist)",
        "license": "CC BY-SA 3.0 (Yann LeCun & Corinna Cortes)",
    },
    "gisette": {
        "url": "https://archive.ics.uci.edu/static/public/170/gisette.zip",
        "citation": "Guyon, Gunn, Ben-Hur, and Dror (2004), UCI ML Repository, DOI 10.24432/C5HP5B",
        "license": "CC BY 4.0",
    },
}


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with urllib.request.urlopen(url, timeout=300) as response:
        path.write_bytes(response.read())


def parse_idx(path):
    with gzip.open(path, "rb") as f:
        buf = f.read()
    ndim = buf[3]
    dims = [int.from_bytes(buf[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(buf, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def save_split(out_dir, name, x_train_raw, y_train, x_test_raw, y_test, feature_names, metadata, preprocessing):
    mean = x_train_raw.mean(axis=0)
    std = x_train_raw.std(axis=0)
    std[std < 1e-8] = 1.0
    x_train = ((x_train_raw - mean) / std).astype(np.float32)
    x_test = ((x_test_raw - mean) / std).astype(np.float32)
    path = out_dir / f"{name}.npz"
    np.savez_compressed(
        path,
        X_train=x_train,
        y_train=y_train.astype(np.float32),
        X_test=x_test,
        y_test=y_test.astype(np.float32),
        feature_names=np.asarray(feature_names, dtype=object),
    )
    meta = {
        **metadata,
        "name": name,
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "n_features_no_intercept": int(x_train.shape[1]),
        "X_train_shape": list(x_train.shape),
        "X_test_shape": list(x_test.shape),
        "preprocessing": preprocessing,
        "processed_path": str(path),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(meta, sort_keys=True))


def process_mnist79(raw_dir, out_dir):
    arrays = {}
    for key, fname in MNIST_FILES.items():
        path = raw_dir / fname
        download(MNIST_BASE + fname, path)
        arrays[key] = parse_idx(path)
    splits = {}
    for split in ["train", "test"]:
        images = arrays[f"{split}_images"]
        labels = arrays[f"{split}_labels"]
        keep = (labels == 7) | (labels == 9)
        x = images[keep].reshape(-1, 784).astype(np.float32) / 255.0
        y = (labels[keep] == 9).astype(np.float32)
        splits[split] = (x, y)
    names = [f"pix_{i:03d}" for i in range(784)]
    preprocessing = (
        "MNIST idx files parsed with numpy (big-endian magic+dims header); kept digits 7 and 9 "
        "(y=1 for digit 9); MNIST's own train/test split; pixels scaled by 1/255 then per-feature "
        "standardized using train-set mean/std with std<1e-8 replaced by 1.0 for zero-variance "
        "pixels; all 784 features kept."
    )
    save_split(out_dir, "mnist79", splits["train"][0], splits["train"][1], splits["test"][0], splits["test"][1], names, DATASETS["mnist79"], preprocessing)


def read_gisette_matrix(zf, member):
    with zf.open(member) as f:
        text = f.read().decode("ascii")
    values = np.array(text.split(), dtype=np.float32)
    return values.reshape(-1, 5000)


def read_gisette_labels(zf, member):
    with zf.open(member) as f:
        raw = np.array(f.read().split(), dtype=np.float32)
    return (raw > 0).astype(np.float32)


def process_gisette(raw_dir, out_dir):
    zip_path = raw_dir / "gisette.zip"
    download(DATASETS["gisette"]["url"], zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        x_train = read_gisette_matrix(zf, "GISETTE/gisette_train.data")
        y_train = read_gisette_labels(zf, "GISETTE/gisette_train.labels")
        x_test = read_gisette_matrix(zf, "GISETTE/gisette_valid.data")
        y_test = read_gisette_labels(zf, "gisette_valid.labels")
    names = [f"gisette_{i:04d}" for i in range(5000)]
    preprocessing = (
        "UCI Gisette (NIPS 2003 feature selection challenge): gisette_train.data (6000x5000) as "
        "train; gisette_valid.data+labels (1000x5000) used as test because official test labels "
        "are withheld; labels mapped from {-1,+1} to {0,1}; per-feature standardized using "
        "train-set mean/std with std<1e-8 replaced by 1.0 for zero-variance features."
    )
    save_split(out_dir, "gisette", x_train, y_train, x_test, y_test, names, DATASETS["gisette"], preprocessing)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist79", "gisette", "all"], default="all")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/real"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = ["mnist79", "gisette"] if args.dataset == "all" else [args.dataset]
    for name in selected:
        if name == "mnist79":
            process_mnist79(args.raw_dir, args.out_dir)
        elif name == "gisette":
            process_gisette(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
