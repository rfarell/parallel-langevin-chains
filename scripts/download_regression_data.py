import argparse
import csv
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np


DATASETS = {
    "superconduct": {
        "url": "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip",
        "csv_name": "train.csv",
        "target": "critical_temp",
        "delimiter": ",",
        "citation": "Hamidieh (2018), UCI ML Repository, DOI 10.24432/C53P47",
        "license": "CC BY 4.0",
    },
    "winequality": {
        "url": "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
        "csv_name": "winequality-white.csv",
        "target": "quality",
        "delimiter": ";",
        "citation": "Cortez, Cerdeira, Almeida, Matos, and Reis (2009), UCI ML Repository, DOI 10.24432/C56S3T",
        "license": "CC BY 4.0",
    },
}


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def split_indices(n, seed=20260504, train_fraction=0.8):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    split = int(round(train_fraction * n))
    return np.sort(perm[:split]), np.sort(perm[split:])


def load_csv_from_zip(zip_path, csv_name, target, delimiter):
    with zipfile.ZipFile(zip_path) as zf:
        member = next(m for m in zf.namelist() if m.endswith(csv_name))
        text = io.TextIOWrapper(zf.open(member), encoding="utf-8")
        reader = csv.reader(text, delimiter=delimiter)
        header = [h.strip().strip('"') for h in next(reader)]
        rows = [[float(v) for v in row] for row in reader if row]
    t = header.index(target)
    data = np.asarray(rows, dtype=np.float64)
    x = np.delete(data, t, axis=1)
    y = data[:, t]
    names = [h for i, h in enumerate(header) if i != t]
    return x, y, names


def save_dataset(out_dir, name, x, y, feature_names, metadata):
    # Standardize features AND target with train-split statistics.
    train_idx, test_idx = split_indices(len(y))
    x_mean = x[train_idx].mean(axis=0)
    x_std = x[train_idx].std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    y_mean = y[train_idx].mean()
    y_std = y[train_idx].std()
    x_s = ((x - x_mean) / x_std).astype(np.float32)
    y_s = ((y - y_mean) / y_std).astype(np.float32)
    path = out_dir / f"{name}.npz"
    np.savez_compressed(
        path,
        X_train=x_s[train_idx],
        y_train=y_s[train_idx],
        X_test=x_s[test_idx],
        y_test=y_s[test_idx],
        feature_names=np.asarray(feature_names, dtype=object),
    )
    meta = {
        "citation": metadata["citation"],
        "license": metadata["license"],
        "url": metadata["url"],
        "target": metadata["target"],
        "target_standardized": True,
        "name": name,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_features_no_intercept": int(x.shape[1]),
        "processed_path": str(path),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(meta, sort_keys=True))


def process(name, raw_dir, out_dir):
    spec = DATASETS[name]
    raw_path = raw_dir / Path(spec["url"]).name.replace("+", "_")
    download(spec["url"], raw_path)
    x, y, names = load_csv_from_zip(raw_path, spec["csv_name"], spec["target"], spec["delimiter"])
    save_dataset(out_dir, name, x, y, names, spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["superconduct", "winequality"], default="superconduct")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/real"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "superconduct":
        try:
            process("superconduct", args.raw_dir, args.out_dir)
            return
        except Exception as exc:
            print(f"superconduct download failed ({exc}); falling back to winequality-white")
    process("winequality", args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
