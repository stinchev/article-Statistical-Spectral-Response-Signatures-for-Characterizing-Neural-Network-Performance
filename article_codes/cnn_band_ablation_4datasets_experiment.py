from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


MODE_LABEL_EN = {"high": "high frequencies", "low": "low frequencies", "mid": "mid frequencies"}
BAND_LABEL_EN = {"low": "low band", "mid": "mid band", "high": "high band"}


DATASET_SPECS = {
    "mnist": {"display_name": "MNIST", "image_shape": (28, 28), "band_low_max": 9.2195, "band_high_min": 13.0},
    "fashion_mnist": {"display_name": "Fashion-MNIST", "image_shape": (28, 28), "band_low_max": 9.2195, "band_high_min": 13.0},
    "kmnist": {"display_name": "KMNIST", "image_shape": (28, 28), "band_low_max": 9.2195, "band_high_min": 13.0},
    "cifar10": {"display_name": "CIFAR-10 (grayscale)", "image_shape": (32, 32), "band_low_max": 10.3, "band_high_min": 14.8},
}
DATASET_ORDER = ["mnist", "fashion_mnist", "kmnist", "cifar10"]


def load_mnist_raw():
    return keras.datasets.mnist.load_data()


def load_fashion_mnist_raw():
    return keras.datasets.fashion_mnist.load_data()


def _load_kmnist_from_npz():
    base_url = "http://codh.rois.ac.jp/kmnist/dataset/kmnist/"
    files = {
        "x_train": "kmnist-train-imgs.npz", "y_train": "kmnist-train-labels.npz",
        "x_test": "kmnist-test-imgs.npz", "y_test": "kmnist-test-labels.npz",
    }
    arrays = {}
    for key, fname in files.items():
        path = keras.utils.get_file(fname, origin=base_url + fname, cache_subdir="datasets/kmnist")
        arrays[key] = np.load(path)["arr_0"]
    return (arrays["x_train"], arrays["y_train"]), (arrays["x_test"], arrays["y_test"])


def load_kmnist_raw():
    try:
        import tensorflow_datasets as tfds
        x_train, y_train = tfds.as_numpy(tfds.load("kmnist", split="train", as_supervised=True, batch_size=-1))
        x_test, y_test = tfds.as_numpy(tfds.load("kmnist", split="test", as_supervised=True, batch_size=-1))
        return (x_train.squeeze(-1), y_train), (x_test.squeeze(-1), y_test)
    except Exception as exc:
        print(f"tensorflow_datasets not available for KMNIST ({exc}); downloading official .npz files...")
        return _load_kmnist_from_npz()


def _robust_cifar10_download(max_retries=10):
    import hashlib
    import tempfile
    import time
    import urllib.request

    origin = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    expected_sha256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"

    if "KERAS_HOME" in os.environ:
        base = os.path.expanduser(os.environ["KERAS_HOME"])
    else:
        base = os.path.expanduser("~/.keras")
    if not os.path.isdir(base) or not os.access(base, os.W_OK):
        base = os.path.join(tempfile.gettempdir(), ".keras")
    datadir = os.path.join(base, "datasets")
    os.makedirs(datadir, exist_ok=True)
    target = os.path.join(datadir, "cifar-10-batches-py-target_archive")

    def sha256_of(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    if os.path.exists(target) and sha256_of(target) == expected_sha256:
        return

    for attempt in range(1, max_retries + 1):
        try:
            resume_from = os.path.getsize(target) if os.path.exists(target) else 0
            req = urllib.request.Request(origin)
            if resume_from:
                req.add_header("Range", f"bytes={resume_from}-")
            with urllib.request.urlopen(req, timeout=60) as resp:
                mode = "ab"
                if resume_from and getattr(resp, "status", 200) != 206:
                    resume_from, mode = 0, "wb"
                with open(target, mode) as out:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
            if sha256_of(target) == expected_sha256:
                return
            print(f"CIFAR-10 archive hash mismatch after attempt {attempt}/{max_retries}, retrying...")
        except Exception as exc:
            print(f"CIFAR-10 download attempt {attempt}/{max_retries} failed ({exc}); retrying in 5s...")
            time.sleep(5)
    raise RuntimeError(f"Could not download a valid CIFAR-10 archive after {max_retries} attempts.")


def robust_cifar10_load_data():
    _robust_cifar10_download()
    return keras.datasets.cifar10.load_data()


def load_cifar10_grayscale_raw():
    (x_train, y_train), (x_test, y_test) = robust_cifar10_load_data()
    y_train, y_test = y_train.squeeze(-1), y_test.squeeze(-1)

    def to_gray(x):
        x = x.astype("float32")
        return 0.2989 * x[..., 0] + 0.5870 * x[..., 1] + 0.1140 * x[..., 2]

    return (to_gray(x_train), y_train), (to_gray(x_test), y_test)


LOADER_BY_DATASET = {
    "mnist": load_mnist_raw, "fashion_mnist": load_fashion_mnist_raw,
    "kmnist": load_kmnist_raw, "cifar10": load_cifar10_grayscale_raw,
}


@dataclass
class ExperimentConfig:
    datasets: list[str] = field(default_factory=lambda: list(DATASET_ORDER))
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    val_per_class: int = 500
    train_per_class: int = -1
    test_per_class: int = -1
    cnn_epochs: int = 16
    batch_size: int = 128
    learning_rate: float = 1e-3
    output_root: str = ""
    ablation_step: float = 0.05
    quick: bool = False


def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="CNN band ablation (progressive sweep + complete removal), 4 datasets.")
    parser.add_argument("--datasets", type=str, nargs="+", default=list(DATASET_ORDER), choices=sorted(DATASET_SPECS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--train-per-class", type=int, default=-1)
    parser.add_argument("--test-per-class", type=int, default=-1)
    parser.add_argument("--cnn-epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--ablation-step", type=float, default=0.05)
    parser.add_argument("--quick", action="store_true")
    args, unknown_args = parser.parse_known_args(argv)
    if unknown_args:
        print(f"Ignoring notebook/kernel arguments: {unknown_args}")

    cfg = ExperimentConfig(
        datasets=args.datasets, seeds=args.seeds, val_per_class=args.val_per_class,
        train_per_class=args.train_per_class, test_per_class=args.test_per_class,
        cnn_epochs=args.cnn_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        output_root=args.output_root, ablation_step=args.ablation_step, quick=args.quick,
    )
    if cfg.quick:
        cfg.seeds = cfg.seeds[:1]
        cfg.val_per_class = min(cfg.val_per_class, 50)
        cfg.train_per_class = 300 if cfg.train_per_class <= 0 else min(cfg.train_per_class, 300)
        cfg.test_per_class = 100 if cfg.test_per_class <= 0 else min(cfg.test_per_class, 100)
        cfg.cnn_epochs = min(cfg.cnn_epochs, 3)
        cfg.ablation_step = max(cfg.ablation_step, 0.25)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def default_output_root() -> str:
    if os.path.exists("/kaggle/working"):
        return "/kaggle/working/cnn_band_ablation_outputs_en"
    return "cnn_band_ablation_experiment/outputs_en"


def make_output_dir(root: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(root or default_output_root()) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def select_multiclass_subset(x: np.ndarray, y: np.ndarray, limit_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for digit in range(10):
        idx = np.where(y == digit)[0]
        rng.shuffle(idx)
        effective_limit = len(idx) if limit_per_class <= 0 else min(limit_per_class, len(idx))
        chosen = idx[:effective_limit]
        xs.append(x[chosen])
        ys.append(y[chosen].astype(np.int32))
    x_out = np.concatenate(xs, axis=0)
    y_out = np.concatenate(ys, axis=0)
    perm = rng.permutation(len(x_out))
    return x_out[perm], y_out[perm]


def load_dataset_three_way(dataset: str, cfg: ExperimentConfig, seed: int) -> dict[str, np.ndarray]:
    (x_train_full, y_train_full), (x_test_full, y_test_full) = LOADER_BY_DATASET[dataset]()
    rng = np.random.default_rng(seed)

    train_idx, val_idx = [], []
    for digit in range(10):
        idx = np.where(y_train_full == digit)[0]
        rng.shuffle(idx)
        n_val = min(cfg.val_per_class, len(idx) - 1)
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    x_train_pool, y_train_pool = x_train_full[train_idx], y_train_full[train_idx]
    x_val, y_val = x_train_full[val_idx], y_train_full[val_idx]

    x_train, y_train = select_multiclass_subset(x_train_pool, y_train_pool, cfg.train_per_class, seed)
    x_test, y_test = select_multiclass_subset(x_test_full, y_test_full, cfg.test_per_class, seed + 1000)

    def prep(x):
        return x.astype("float32") / 255.0

    x_train, x_val, x_test = prep(x_train), prep(x_val), prep(x_test)
    return {
        "x_train_img": x_train, "x_val_img": x_val, "x_test_img": x_test,
        "y_train": y_train.astype("int32"), "y_val": y_val.astype("int32"), "y_test": y_test.astype("int32"),
        "n_train": len(x_train), "n_val": len(x_val), "n_test": len(x_test),
    }


def build_cnn(learning_rate: float, image_shape: tuple[int, int]) -> keras.Model:
    height, width = image_shape
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(height, width, 1)),
            keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(10, activation="softmax"),
        ],
        name="vision10_cnn",
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def fit_with_early_stopping(model, x_train, y_train, x_val, y_val, epochs, batch_size, monitor):
    callbacks = [
        keras.callbacks.EarlyStopping(monitor=monitor, patience=max(2, epochs // 4), restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=max(2, epochs // 6), min_lr=1e-5),
    ]
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, verbose=2, callbacks=callbacks)
    return history, len(history.history["loss"])


def radial_distance_map(height: int, width: int) -> np.ndarray:
    yy, xx = np.indices((height, width))
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    return np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)


def radial_keep_mask(height: int, width: int, remove_fraction: float) -> np.ndarray:
    if remove_fraction <= 0:
        return np.ones((height, width), dtype=bool)
    radius = radial_distance_map(height, width)
    flat_order = np.argsort(radius, axis=None)
    keep_count = max(1, int(round((1.0 - remove_fraction) * height * width)))
    keep_mask = np.zeros(height * width, dtype=bool)
    keep_mask[flat_order[:keep_count]] = True
    return keep_mask.reshape(height, width)


def low_frequency_keep_mask(height: int, width: int, remove_fraction: float) -> np.ndarray:
    radius = radial_distance_map(height, width)
    flat_order = np.argsort(radius, axis=None)
    remove_count = min(max(int(round(remove_fraction * height * width)), 0), height * width - 1)
    keep_mask = np.ones(height * width, dtype=bool)
    if remove_count > 0:
        keep_mask[flat_order[:remove_count]] = False
    return keep_mask.reshape(height, width)


def mid_frequency_keep_mask(height: int, width: int, remove_fraction: float, band_mid_rho: float) -> np.ndarray:
    radius = radial_distance_map(height, width)
    distance_to_mid_center = np.abs(radius - band_mid_rho)
    flat_order = np.argsort(distance_to_mid_center, axis=None)
    remove_count = min(max(int(round(remove_fraction * height * width)), 0), height * width - 1)
    keep_mask = np.ones(height * width, dtype=bool)
    if remove_count > 0:
        keep_mask[flat_order[:remove_count]] = False
    return keep_mask.reshape(height, width)


def remove_high_frequencies(images: np.ndarray, remove_fraction: float):
    mask = radial_keep_mask(images.shape[1], images.shape[2], remove_fraction)
    shifted = np.fft.fftshift(np.fft.fft2(images, axes=(-2, -1)), axes=(-2, -1))
    restored = np.fft.ifft2(np.fft.ifftshift(shifted * mask[np.newaxis], axes=(-2, -1)), axes=(-2, -1)).real
    return np.clip(restored, 0.0, 1.0), mask


def remove_low_frequencies(images: np.ndarray, remove_fraction: float):
    mask = low_frequency_keep_mask(images.shape[1], images.shape[2], remove_fraction)
    shifted = np.fft.fftshift(np.fft.fft2(images, axes=(-2, -1)), axes=(-2, -1))
    restored = np.fft.ifft2(np.fft.ifftshift(shifted * mask[np.newaxis], axes=(-2, -1)), axes=(-2, -1)).real
    return np.clip(restored, 0.0, 1.0), mask


def remove_mid_frequencies(images: np.ndarray, remove_fraction: float, band_mid_rho: float):
    mask = mid_frequency_keep_mask(images.shape[1], images.shape[2], remove_fraction, band_mid_rho)
    shifted = np.fft.fftshift(np.fft.fft2(images, axes=(-2, -1)), axes=(-2, -1))
    restored = np.fft.ifft2(np.fft.ifftshift(shifted * mask[np.newaxis], axes=(-2, -1)), axes=(-2, -1)).real
    return np.clip(restored, 0.0, 1.0), mask


def radial_frequency_grid_centered(height: int, width: int) -> np.ndarray:
    yy, xx = np.indices((height, width))
    cy, cx = height // 2, width // 2
    return np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)


def band_keep_mask(height: int, width: int, band: str, band_low_max: float, band_high_min: float) -> np.ndarray:
    radius = radial_frequency_grid_centered(height, width)
    if band == "low":
        remove = radius <= band_low_max
    elif band == "mid":
        remove = (radius > band_low_max) & (radius <= band_high_min)
    elif band == "high":
        remove = radius > band_high_min
    else:
        raise ValueError(f"unknown band: {band}")
    return ~remove


def remove_band(images: np.ndarray, band: str, band_low_max: float, band_high_min: float):
    mask = band_keep_mask(images.shape[1], images.shape[2], band, band_low_max, band_high_min)
    shifted = np.fft.fftshift(np.fft.fft2(images, axes=(-2, -1)), axes=(-2, -1))
    restored = np.fft.ifft2(np.fft.ifftshift(shifted * mask[np.newaxis], axes=(-2, -1)), axes=(-2, -1)).real
    return np.clip(restored, 0.0, 1.0), mask


def mask_energy_stats(images: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    total = int(mask.size)
    kept = int(mask.sum())
    shifted = np.fft.fftshift(np.fft.fft2(images, axes=(-2, -1)), axes=(-2, -1))
    energy = np.abs(shifted) ** 2
    retained = energy[:, mask].sum(axis=1)
    total_energy = energy.reshape(len(images), -1).sum(axis=1)
    retained_fraction = float(np.mean(retained / np.maximum(total_energy, 1e-12)))
    return {"kept_coefficients": kept, "removed_coefficients": total - kept,
            "retained_energy_fraction": retained_fraction, "removed_energy_fraction": 1.0 - retained_fraction}


def evaluate_band_ablation(model, x_test_img, y_test, band_low_max, band_high_min) -> list[dict]:
    baseline_loss, baseline_acc = model.evaluate(x_test_img[..., np.newaxis], y_test, verbose=0)
    rows = []
    for band in ("low", "mid", "high"):
        degraded, mask = remove_band(x_test_img, band, band_low_max, band_high_min)
        loss, accuracy = model.evaluate(degraded[..., np.newaxis], y_test, verbose=0)
        row = {"band": band, "loss": float(loss), "accuracy": float(accuracy)}
        row.update(mask_energy_stats(x_test_img, mask))
        row["accuracy_drop_from_baseline_pp"] = 100.0 * (baseline_acc - accuracy)
        rows.append(row)
    return rows


REMOVE_FN_BY_MODE = {"high": remove_high_frequencies, "low": remove_low_frequencies}


def evaluate_ablation(model, x_test_img, y_test, ablation_step, mode, band_mid_rho):
    rows = []
    for remove_fraction in np.arange(0.0, 0.951, ablation_step):
        remove_fraction = float(remove_fraction)
        if mode == "mid":
            degraded, mask = remove_mid_frequencies(x_test_img, remove_fraction, band_mid_rho)
        else:
            degraded, mask = REMOVE_FN_BY_MODE[mode](x_test_img, remove_fraction)
        loss, accuracy = model.evaluate(degraded[..., np.newaxis], y_test, verbose=0)
        row = {"removed_fraction_key": remove_fraction, "mode": mode, "loss": float(loss), "accuracy": float(accuracy)}
        row.update(mask_energy_stats(x_test_img, mask))
        rows.append(row)
    baseline = rows[0]["accuracy"]
    prev = baseline
    for idx, row in enumerate(rows):
        row["accuracy_drop_from_baseline_pp"] = 100.0 * (baseline - row["accuracy"])
        row["incremental_accuracy_drop_pp"] = 0.0 if idx == 0 else 100.0 * (prev - row["accuracy"])
        prev = row["accuracy"]
    return rows


def run_dataset(dataset: str, cfg: ExperimentConfig) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset]
    height, width = spec["image_shape"]
    band_low_max, band_high_min = spec["band_low_max"], spec["band_high_min"]
    band_mid_rho = (band_low_max + band_high_min) / 2.0

    per_seed_results = []
    for seed in cfg.seeds:
        set_seed(seed)
        data = load_dataset_three_way(dataset, cfg, seed)
        print(f"  [seed={seed}] train={data['n_train']} val={data['n_val']} test={data['n_test']}")

        cnn = build_cnn(cfg.learning_rate, (height, width))
        fit_with_early_stopping(
            cnn, data["x_train_img"][..., np.newaxis], data["y_train"],
            data["x_val_img"][..., np.newaxis], data["y_val"],
            cfg.cnn_epochs, cfg.batch_size, "val_accuracy",
        )
        cnn_loss, cnn_accuracy = cnn.evaluate(data["x_test_img"][..., np.newaxis], data["y_test"], verbose=0)
        print(f"  [seed={seed}] baseline accuracy = {cnn_accuracy*100:.2f}%")

        high_rows = evaluate_ablation(cnn, data["x_test_img"], data["y_test"], cfg.ablation_step, "high", band_mid_rho)
        low_rows = evaluate_ablation(cnn, data["x_test_img"], data["y_test"], cfg.ablation_step, "low", band_mid_rho)
        mid_rows = evaluate_ablation(cnn, data["x_test_img"], data["y_test"], cfg.ablation_step, "mid", band_mid_rho)
        band_rows = evaluate_band_ablation(cnn, data["x_test_img"], data["y_test"], band_low_max, band_high_min)

        per_seed_results.append({
            "seed": seed, "n_train": data["n_train"], "n_val": data["n_val"], "n_test": data["n_test"],
            "test_accuracy": float(cnn_accuracy), "test_loss": float(cnn_loss),
            "high_frequency_ablation": high_rows, "low_frequency_ablation": low_rows,
            "mid_frequency_ablation": mid_rows, "band_ablation": band_rows,
        })

    agg = aggregate_seeds(per_seed_results)
    return {
        "dataset": dataset, "display_name": spec["display_name"], "image_shape": [height, width],
        "band_low_max": band_low_max, "band_high_min": band_high_min,
        "per_seed_results": per_seed_results, "aggregate": agg,
    }


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "values": [float(v) for v in arr]}


def aggregate_seeds(per_seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, Any] = {"n_seeds": len(per_seed_results), "seeds": [r["seed"] for r in per_seed_results]}
    agg["cnn_test_accuracy"] = mean_std([r["test_accuracy"] for r in per_seed_results])
    agg["cnn_band_ablation"] = {}
    for band_idx, band in enumerate(("low", "mid", "high")):
        agg["cnn_band_ablation"][band] = {
            "accuracy": mean_std([r["band_ablation"][band_idx]["accuracy"] for r in per_seed_results]),
            "accuracy_drop_from_baseline_pp": mean_std(
                [r["band_ablation"][band_idx]["accuracy_drop_from_baseline_pp"] for r in per_seed_results]
            ),
            "kept_coefficients": per_seed_results[0]["band_ablation"][band_idx]["kept_coefficients"],
            "removed_coefficients": per_seed_results[0]["band_ablation"][band_idx]["removed_coefficients"],
        }
    return agg


def save_ablation_plot(rows_per_seed: list[list[dict]], output_path: Path, mode: str, display_name: str) -> None:
    plt.figure(figsize=(8, 5))
    for seed_idx, rows in enumerate(rows_per_seed):
        x = [100.0 * row["removed_fraction_key"] for row in rows]
        y = [100.0 * row["accuracy"] for row in rows]
        plt.plot(x, y, marker="o", alpha=0.6, label=f"seed {seed_idx}")
    plt.xlabel(f"{MODE_LABEL_EN[mode].capitalize()} coefficients removed (%) — {display_name}")
    plt.ylabel("CNN test accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_band_ablation_plot(band_rows_per_seed: list[list[dict]], output_path: Path, display_name: str) -> None:
    bands = ("low", "mid", "high")
    means, stds = [], []
    for band_idx in range(3):
        values = [100.0 * rows[band_idx]["accuracy"] for rows in band_rows_per_seed]
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)))
    plt.figure(figsize=(6, 5))
    plt.bar([BAND_LABEL_EN[b].capitalize() for b in bands], means, yerr=stds, capsize=6, color=["#4c72b0", "#55a868", "#c44e52"])
    plt.ylabel("CNN test accuracy (%)")
    plt.xlabel(display_name)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def build_report(output_dir: Path, cfg: ExperimentConfig, results: list[dict[str, Any]]) -> None:
    lines = [
        "# CNN band ablation (progressive sweep + complete removal), 4 datasets",
        "",
        "## Setup",
        f"- Seeds: `{cfg.seeds}`",
        f"- Progressive sweep step: `{cfg.ablation_step*100:.0f}%`",
        "",
    ]
    for result in results:
        agg = result["aggregate"]
        lines += [
            f"## {result['display_name']}",
            f"- Baseline accuracy: `{agg['cnn_test_accuracy']['mean']*100:.2f}% ± {agg['cnn_test_accuracy']['std']*100:.2f}pp`",
            "",
            "| Band removed | Modes removed | Accuracy | Drop |",
            "|---|---|---|---|",
        ]
        for band in ("low", "mid", "high"):
            band_res = agg["cnn_band_ablation"][band]
            lines.append(
                f"| {band.capitalize()} | {band_res['removed_coefficients']} | "
                f"{band_res['accuracy']['mean']*100:.2f}% ± {band_res['accuracy']['std']*100:.2f}pp | "
                f"{band_res['accuracy_drop_from_baseline_pp']['mean']:.2f}pp |"
            )
        lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = parse_args()
    output_dir = make_output_dir(cfg.output_root)
    print(f"Datasets: {cfg.datasets}")
    print(f"Output directory: {output_dir.resolve()}")

    results = []
    for dataset in cfg.datasets:
        print(f"\n===== DATASET: {DATASET_SPECS[dataset]['display_name']} =====")
        result = run_dataset(dataset, cfg)
        results.append(result)

        save_ablation_plot([r["high_frequency_ablation"] for r in result["per_seed_results"]],
                            output_dir / f"{dataset}_high_ablation_curve.png", "high", result["display_name"])
        save_ablation_plot([r["low_frequency_ablation"] for r in result["per_seed_results"]],
                            output_dir / f"{dataset}_low_ablation_curve.png", "low", result["display_name"])
        save_ablation_plot([r["mid_frequency_ablation"] for r in result["per_seed_results"]],
                            output_dir / f"{dataset}_mid_ablation_curve.png", "mid", result["display_name"])
        save_band_ablation_plot([r["band_ablation"] for r in result["per_seed_results"]],
                                 output_dir / f"{dataset}_band_ablation_bar.png", result["display_name"])

        with (output_dir / f"{dataset}_metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

    build_report(output_dir, cfg, results)
    with (output_dir / "metrics_all_datasets.json").open("w", encoding="utf-8") as fh:
        json.dump({"config": asdict(cfg), "results": results}, fh, indent=2)

    print("\n=== Aggregate summary ===")
    for result in results:
        agg = result["aggregate"]
        print(f"{result['display_name']}: baseline_accuracy={agg['cnn_test_accuracy']['mean']*100:.2f}%±{agg['cnn_test_accuracy']['std']*100:.2f}pp")
        for band in ("low", "mid", "high"):
            band_res = agg["cnn_band_ablation"][band]
            print(f"  {band} band: accuracy={band_res['accuracy']['mean']*100:.2f}%±{band_res['accuracy']['std']*100:.2f}pp "
                  f"drop={band_res['accuracy_drop_from_baseline_pp']['mean']:.2f}pp")
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
