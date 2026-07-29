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


_trapz = getattr(np, "trapezoid", None) or np.trapz


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


DATASET_SPECS = {
    "mnist": {"display_name": "MNIST", "image_shape": (28, 28), "loader": load_mnist_raw},
    "fashion_mnist": {"display_name": "Fashion-MNIST", "image_shape": (28, 28), "loader": load_fashion_mnist_raw},
    "kmnist": {"display_name": "KMNIST", "image_shape": (28, 28), "loader": load_kmnist_raw},
    "cifar10": {"display_name": "CIFAR-10 (grayscale)", "image_shape": (32, 32), "loader": load_cifar10_grayscale_raw},
}
DATASET_ORDER = ["mnist", "fashion_mnist", "kmnist", "cifar10"]


@dataclass
class ExperimentConfig:
    datasets: list[str] = field(default_factory=lambda: list(DATASET_ORDER))
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    n_points: int = 150
    val_per_class: int = 500
    train_per_class: int = -1
    test_per_class: int = -1
    cnn_epochs: int = 16
    batch_size: int = 128
    learning_rate: float = 1e-3
    output_root: str = ""
    quick: bool = False


def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Cumulative spectral sweep + CDF, 4 datasets, in one run.")
    parser.add_argument("--datasets", type=str, nargs="+", default=list(DATASET_ORDER), choices=sorted(DATASET_SPECS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--n-points", type=int, default=150)
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--train-per-class", type=int, default=-1)
    parser.add_argument("--test-per-class", type=int, default=-1)
    parser.add_argument("--cnn-epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--quick", action="store_true")
    args, unknown_args = parser.parse_known_args(argv)
    if unknown_args:
        print(f"Ignoring notebook/kernel arguments: {unknown_args}")

    cfg = ExperimentConfig(
        datasets=args.datasets, seeds=args.seeds, n_points=args.n_points,
        val_per_class=args.val_per_class, train_per_class=args.train_per_class, test_per_class=args.test_per_class,
        cnn_epochs=args.cnn_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        output_root=args.output_root, quick=args.quick,
    )
    if cfg.quick:
        cfg.seeds = cfg.seeds[:1]
        cfg.n_points = min(cfg.n_points, 20)
        cfg.val_per_class = min(cfg.val_per_class, 50)
        cfg.train_per_class = 300 if cfg.train_per_class <= 0 else min(cfg.train_per_class, 300)
        cfg.test_per_class = 100 if cfg.test_per_class <= 0 else min(cfg.test_per_class, 100)
        cfg.cnn_epochs = min(cfg.cnn_epochs, 3)
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
        return "/kaggle/working/spectral_cdf_outputs_en"
    return "spectral_cdf_experiment/outputs_en"


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
    (x_train_full, y_train_full), (x_test_full, y_test_full) = DATASET_SPECS[dataset]["loader"]()
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


def radial_frequency_grid_centered(height: int, width: int) -> np.ndarray:
    yy, xx = np.indices((height, width))
    cy, cx = height // 2, width // 2
    return np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)


def build_keep_count_schedule(total_modes: int, n_points: int) -> np.ndarray:
    raw = np.linspace(1, total_modes, min(n_points, total_modes))
    counts = np.unique(np.round(raw).astype(int))
    return np.clip(counts, 1, total_modes)


def reconstruct_with_keep_count(images: np.ndarray, order: np.ndarray, height: int, width: int, n_keep: int) -> np.ndarray:
    keep_flat = np.zeros(height * width, dtype=bool)
    keep_flat[order[:n_keep]] = True
    keep_mask = keep_flat.reshape(height, width)
    shifted = np.fft.fftshift(np.fft.fft2(images, axes=(-2, -1)), axes=(-2, -1))
    restored = np.fft.ifft2(np.fft.ifftshift(shifted * keep_mask[np.newaxis], axes=(-2, -1)), axes=(-2, -1)).real
    return np.clip(restored, 0.0, 1.0)


def sweep_cumulative_lowpass(model, x_test_img: np.ndarray, y_test: np.ndarray, height: int, width: int,
                              n_points: int, batch_size: int) -> list[dict[str, float]]:
    radius = radial_frequency_grid_centered(height, width)
    order = np.argsort(radius, axis=None)
    sorted_radius = radius.flatten()[order]
    total_modes = height * width
    rho_max = float(sorted_radius[-1])
    counts = build_keep_count_schedule(total_modes, n_points)

    rows = []
    for n_keep in counts:
        degraded = reconstruct_with_keep_count(x_test_img, order, height, width, int(n_keep))
        _, accuracy = model.evaluate(degraded[..., np.newaxis], y_test, batch_size=batch_size, verbose=0)
        radius_cutoff = float(sorted_radius[n_keep - 1])
        rows.append({
            "modes_kept": int(n_keep),
            "radius_cutoff": radius_cutoff,
            "normalized_radius": radius_cutoff / rho_max,
            "accuracy": float(accuracy),
        })
    return rows


def pool_adjacent_violators(y: np.ndarray) -> np.ndarray:
    values: list[float] = []
    weights: list[float] = []
    for yi in y:
        values.append(float(yi))
        weights.append(1.0)
        while len(values) > 1 and values[-2] > values[-1]:
            merged_weight = weights[-2] + weights[-1]
            merged_value = (values[-2] * weights[-2] + values[-1] * weights[-1]) / merged_weight
            values.pop()
            weights.pop()
            values[-1] = merged_value
            weights[-1] = merged_weight
    fitted = np.empty(len(y), dtype=float)
    pos = 0
    for value, weight in zip(values, weights):
        n = int(round(weight))
        fitted[pos:pos + n] = value
        pos += n
    return fitted


def fit_isotonic_cdf(normalized_radius: np.ndarray, normalized_accuracy: np.ndarray) -> np.ndarray:
    y = np.clip(np.asarray(normalized_accuracy, dtype=float), 0.0, 1.0)
    return np.clip(pool_adjacent_violators(y), 0.0, 1.0)


def cdf_statistics(normalized_radius: np.ndarray, cdf_values: np.ndarray) -> dict[str, float]:
    x = np.asarray(normalized_radius, dtype=float)
    f = np.asarray(cdf_values, dtype=float)
    if x[0] > 0.0:
        x = np.concatenate([[0.0], x])
        f = np.concatenate([[0.0], f])
    if x[-1] < 1.0:
        x = np.concatenate([x, [1.0]])
        f = np.concatenate([f, [1.0]])
    survival = 1.0 - f
    mean = float(_trapz(survival, x))
    second_moment = float(2.0 * _trapz(x * survival, x))
    variance = max(second_moment - mean ** 2, 0.0)
    std = float(np.sqrt(variance))
    median = float(np.interp(0.5, f, x))
    density = np.diff(f) / np.diff(x)
    mode_idx = int(np.argmax(density))
    mode = float(0.5 * (x[mode_idx] + x[mode_idx + 1]))
    return {"mean": mean, "std": std, "median": median, "mode": mode}


def run_dataset(dataset: str, cfg: ExperimentConfig) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset]
    height, width = spec["image_shape"]
    per_seed_rows: list[list[dict]] = []
    per_seed_iso: list[np.ndarray] = []
    per_seed_stats: list[dict[str, float]] = []
    baseline_accs: list[float] = []

    for seed in cfg.seeds:
        print(f"  [seed={seed}] training CNN...")
        set_seed(seed)
        data = load_dataset_three_way(dataset, cfg, seed)
        cnn = build_cnn(cfg.learning_rate, (height, width))
        fit_with_early_stopping(
            cnn, data["x_train_img"][..., np.newaxis], data["y_train"],
            data["x_val_img"][..., np.newaxis], data["y_val"],
            cfg.cnn_epochs, cfg.batch_size, "val_accuracy",
        )
        baseline_loss, baseline_acc = cnn.evaluate(data["x_test_img"][..., np.newaxis], data["y_test"], verbose=0)
        print(f"  [seed={seed}] baseline accuracy (untouched) = {baseline_acc*100:.2f}% ; sweeping spectrum...")

        rows = sweep_cumulative_lowpass(cnn, data["x_test_img"], data["y_test"], height, width, cfg.n_points, cfg.batch_size)
        accuracy_full = rows[-1]["accuracy"]
        for row in rows:
            row["normalized_accuracy"] = float(np.clip(row["accuracy"] / accuracy_full, 0.0, 1.0))

        x = np.array([r["normalized_radius"] for r in rows])
        y = np.array([r["normalized_accuracy"] for r in rows])
        f_iso = fit_isotonic_cdf(x, y)
        stats = cdf_statistics(x, f_iso)

        per_seed_rows.append(rows)
        per_seed_iso.append(f_iso)
        per_seed_stats.append(stats)
        baseline_accs.append(float(baseline_acc))

    agg_stats: dict[str, dict[str, float]] = {}
    for key in ("mean", "std", "median", "mode"):
        values = [s[key] for s in per_seed_stats]
        agg_stats[key] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}
    agg_stats["baseline_accuracy"] = {
        "mean": float(np.mean(baseline_accs)), "std": float(np.std(baseline_accs)), "values": baseline_accs
    }

    return {
        "dataset": dataset, "display_name": spec["display_name"], "image_shape": [height, width],
        "per_seed_rows": per_seed_rows,
        "per_seed_isotonic_cdf": [f.tolist() for f in per_seed_iso],
        "per_seed_stats": per_seed_stats,
        "aggregate_stats": agg_stats,
    }


def renormalize_srf(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y0 = y[0]
    denom = max(1.0 - y0, 1e-12)
    return np.clip((y - y0) / denom, 0.0, 1.0)


def save_dataset_curve_plot(result: dict[str, Any], output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for seed_idx, rows in enumerate(result["per_seed_rows"]):
        x = [r["normalized_radius"] for r in rows]
        y_iso = np.array(result["per_seed_isotonic_cdf"][seed_idx])
        y_srf = renormalize_srf(y_iso)
        y_raw_srf = renormalize_srf(np.array([r["normalized_accuracy"] for r in rows]))
        plt.scatter(x, y_raw_srf, s=10, alpha=0.35, color=f"C{seed_idx}")
        plt.plot(x, y_srf, color=f"C{seed_idx}", label=f"seed {seed_idx}")
    plt.xlabel("Normalized spectral radius (r = ρ/ρ_max)")
    plt.ylabel("SRF(r)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_comparison_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for idx, result in enumerate(results):
        rows0 = result["per_seed_rows"][0]
        x = [r["normalized_radius"] for r in rows0]
        srf_per_seed = np.array([renormalize_srf(np.array(c)) for c in result["per_seed_isotonic_cdf"]])
        mean_srf = srf_per_seed.mean(axis=0)
        plt.plot(x, mean_srf, label=result["display_name"], color=f"C{idx}")
    plt.xlabel("Normalized spectral radius (r = ρ/ρ_max)")
    plt.ylabel("Mean SRF(r)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def build_report(output_dir: Path, cfg: ExperimentConfig, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Cumulative spectral sweep + CDF (MNIST / Fashion-MNIST / KMNIST / grayscale CIFAR-10)",
        "",
        "## Setup",
        f"- Seeds: `{cfg.seeds}`",
        f"- Points per curve: `{cfg.n_points}`",
        "",
        "## Statistics of the implicit distribution (normalized spectral radius, mean ± std across seeds)",
        "",
        "| Dataset | Baseline accuracy | E[r] (mean) | Std. dev. | Median | Mode |",
        "|---|---|---|---|---|---|",
    ]
    for result in results:
        agg = result["aggregate_stats"]
        lines.append(
            f"| {result['display_name']} "
            f"| {agg['baseline_accuracy']['mean']*100:.2f}% ± {agg['baseline_accuracy']['std']*100:.2f}pp "
            f"| {agg['mean']['mean']:.4f} ± {agg['mean']['std']:.4f} "
            f"| {agg['std']['mean']:.4f} ± {agg['std']['std']:.4f} "
            f"| {agg['median']['mean']:.4f} ± {agg['median']['std']:.4f} "
            f"| {agg['mode']['mean']:.4f} ± {agg['mode']['std']:.4f} |"
        )
    lines += [
        "",
        "`E[r]`, `Std. dev.`, `Median` and `Mode` are statistics of the variable "
        "\"normalized spectral radius required to classify correctly\" (r ∈ [0,1]), "
        "computed from the empirical CDF (normalized accuracy vs. normalized radius) "
        "after fitting it to a monotone curve via isotonic regression.",
        "",
    ]
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
        save_dataset_curve_plot(result, output_dir / f"{dataset}_spectral_cdf.png")

        with (output_dir / f"{dataset}_metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

    save_comparison_plot(results, output_dir / "comparison_spectral_cdf.png")
    build_report(output_dir, cfg, results)

    with (output_dir / "metrics_all_datasets.json").open("w", encoding="utf-8") as fh:
        json.dump({"config": asdict(cfg), "results": results}, fh, indent=2)

    print("\n=== Aggregate summary ===")
    for result in results:
        agg = result["aggregate_stats"]
        print(
            f"{result['display_name']}: baseline_accuracy={agg['baseline_accuracy']['mean']*100:.2f}%±{agg['baseline_accuracy']['std']*100:.2f}pp "
            f"E[r]={agg['mean']['mean']:.4f}±{agg['mean']['std']:.4f} "
            f"std[r]={agg['std']['mean']:.4f}±{agg['std']['std']:.4f} "
            f"median={agg['median']['mean']:.4f}±{agg['median']['std']:.4f} "
            f"mode={agg['mode']['mean']:.4f}±{agg['mode']['std']:.4f}"
        )
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
