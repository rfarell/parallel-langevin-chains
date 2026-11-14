import argparse
import csv
import json
import urllib.request
from pathlib import Path

import numpy as np


DATASETS = {
    "spambase": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
        "citation": "Hopkins, Reeber, Forman, and Suermondt (1999), UCI ML Repository, DOI 10.24432/C53G6X",
        "license": "CC BY 4.0",
    },
    "wdbc": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
        "citation": "Wolberg, Mangasarian, Street, and Street (1993), UCI ML Repository, DOI 10.24432/C5DW2B",
        "license": "CC BY 4.0",
    },
    "adult": {
        "urls": [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        ],
        "citation": "Becker and Kohavi (1996), UCI ML Repository, DOI 10.24432/C5XW20",
        "license": "CC BY 4.0",
    },
}


ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]
ADULT_NUMERIC = {"age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"}


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def standardize_train_test(x, split):
    train = x[:split]
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-8] = 1.0
    return ((x - mean) / std).astype(np.float32)


def split_indices(n, seed=20260504, train_fraction=0.8):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    split = int(round(train_fraction * n))
    train_idx = np.sort(perm[:split])
    test_idx = np.sort(perm[split:])
    return train_idx, test_idx


def save_dataset(out_dir, name, x, y, feature_names, metadata, fixed_split=None):
    if fixed_split is None:
        train_idx, test_idx = split_indices(len(y))
    else:
        train_idx, test_idx = fixed_split
    x_train_raw = x[train_idx]
    mean = x_train_raw.mean(axis=0)
    std = x_train_raw.std(axis=0)
    std[std < 1e-8] = 1.0
    x_std = ((x - mean) / std).astype(np.float32)
    path = out_dir / f"{name}.npz"
    np.savez_compressed(
        path,
        X_train=x_std[train_idx],
        y_train=y[train_idx].astype(np.float32),
        X_test=x_std[test_idx],
        y_test=y[test_idx].astype(np.float32),
        feature_names=np.asarray(feature_names, dtype=object),
    )
    meta = {
        **metadata,
        "name": name,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_features_no_intercept": int(x.shape[1]),
        "processed_path": str(path),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(meta, sort_keys=True))


def process_spambase(raw_dir, out_dir):
    url = DATASETS["spambase"]["url"]
    raw_path = raw_dir / "spambase.data"
    download(url, raw_path)
    data = np.loadtxt(raw_path, delimiter=",", dtype=np.float32)
    x = data[:, :-1]
    y = data[:, -1]
    names = [f"spambase_{i:02d}" for i in range(x.shape[1])]
    save_dataset(out_dir, "spambase", x, y, names, DATASETS["spambase"])


def process_wdbc(raw_dir, out_dir):
    url = DATASETS["wdbc"]["url"]
    raw_path = raw_dir / "wdbc.data"
    download(url, raw_path)
    rows = []
    with raw_path.open("r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row:
                rows.append(row)
    y = np.asarray([1.0 if row[1] == "M" else 0.0 for row in rows], dtype=np.float32)
    x = np.asarray([[float(v) for v in row[2:]] for row in rows], dtype=np.float32)
    names = [f"wdbc_{i:02d}" for i in range(x.shape[1])]
    save_dataset(out_dir, "wdbc", x, y, names, DATASETS["wdbc"])


def parse_adult_file(path, is_test):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("|"):
                continue
            values = [v.strip() for v in line.split(",")]
            if len(values) != len(ADULT_COLUMNS) or "?" in values:
                continue
            if is_test:
                values[-1] = values[-1].rstrip(".")
            rows.append(dict(zip(ADULT_COLUMNS, values)))
    return rows


def adult_matrix(rows, categories=None):
    if categories is None:
        categories = {}
        for col in ADULT_COLUMNS[:-1]:
            if col not in ADULT_NUMERIC:
                categories[col] = sorted({row[col] for row in rows})
    numeric_cols = [col for col in ADULT_COLUMNS[:-1] if col in ADULT_NUMERIC]
    cat_cols = [col for col in ADULT_COLUMNS[:-1] if col not in ADULT_NUMERIC]
    feature_names = list(numeric_cols)
    for col in cat_cols:
        feature_names.extend(f"{col}={value}" for value in categories[col])
    x = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    for i, row in enumerate(rows):
        cursor = 0
        for col in numeric_cols:
            x[i, cursor] = float(row[col])
            cursor += 1
        for col in cat_cols:
            values = categories[col]
            try:
                offset = values.index(row[col])
            except ValueError:
                offset = -1
            if offset >= 0:
                x[i, cursor + offset] = 1.0
            cursor += len(values)
    y = np.asarray([1.0 if row["income"] == ">50K" else 0.0 for row in rows], dtype=np.float32)
    return x, y, feature_names, categories


def process_adult(raw_dir, out_dir):
    raw_paths = [raw_dir / "adult.data", raw_dir / "adult.test"]
    for url, path in zip(DATASETS["adult"]["urls"], raw_paths):
        download(url, path)
    train_rows = parse_adult_file(raw_paths[0], is_test=False)
    test_rows = parse_adult_file(raw_paths[1], is_test=True)
    x_train, y_train, feature_names, categories = adult_matrix(train_rows)
    x_test, y_test, _, _ = adult_matrix(test_rows, categories=categories)
    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])
    split = len(y_train)
    save_dataset(
        out_dir,
        "adult",
        x,
        y,
        feature_names,
        DATASETS["adult"],
        fixed_split=(np.arange(split), np.arange(split, len(y))),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["spambase", "wdbc", "adult", "all"], default="all")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/real"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = ["spambase", "wdbc", "adult"] if args.dataset == "all" else [args.dataset]
    for name in selected:
        if name == "spambase":
            process_spambase(args.raw_dir, args.out_dir)
        elif name == "wdbc":
            process_wdbc(args.raw_dir, args.out_dir)
        elif name == "adult":
            process_adult(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
