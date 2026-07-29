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
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import torchvision

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

ARCH_ORDER = ["cnn", "resnet18"]
ARCH_DISPLAY = {"cnn": "CNN (from scratch)", "resnet18": "ResNet-18 (pretrained, frozen)"}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass
class ExperimentConfig:
    dataset: str = "mnist"
    architectures: list[str] = field(default_factory=lambda: list(ARCH_ORDER))
    seeds: list[int] = field(default_factory=lambda: list(range(42, 52)))
    n_points: int = 150
    val_per_class: int = 500
    train_per_class: int = -1
    test_per_class: int = -1

    cnn_epochs: int = 16
    cnn_batch_size: int = 128
    cnn_learning_rate: float = 1e-3

    resnet_head_epochs: int = 30
    resnet_patience: int = 5
    resnet_batch_size: int = 128
    resnet_feature_batch_size: int = 256
    resnet_learning_rate: float = 1e-3
    eval_subset_size: int = 2000
    output_root: str = ""
    quick: bool = False


def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="SRF stability across seeds, CNN and ResNet-18, a single dataset.")
    parser.add_argument("--dataset", type=str, default="mnist", choices=sorted(DATASET_SPECS.keys()))
    parser.add_argument("--architectures", type=str, nargs="+", default=list(ARCH_ORDER), choices=ARCH_ORDER)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--n-points", type=int, default=150)
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--train-per-class", type=int, default=-1)
    parser.add_argument("--test-per-class", type=int, default=-1)
    parser.add_argument("--cnn-epochs", type=int, default=16)
    parser.add_argument("--cnn-batch-size", type=int, default=128)
    parser.add_argument("--cnn-learning-rate", type=float, default=1e-3)
    parser.add_argument("--resnet-head-epochs", type=int, default=30)
    parser.add_argument("--resnet-patience", type=int, default=5)
    parser.add_argument("--resnet-batch-size", type=int, default=128)
    parser.add_argument("--resnet-feature-batch-size", type=int, default=256)
    parser.add_argument("--resnet-learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-subset-size", type=int, default=2000, help="-1 to use the full test set in the ResNet-18 sweep")
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--quick", action="store_true")
    args, unknown_args = parser.parse_known_args(argv)
    if unknown_args:
        print(f"Ignoring notebook/kernel arguments: {unknown_args}")

    cfg = ExperimentConfig(
        dataset=args.dataset, architectures=args.architectures, seeds=args.seeds, n_points=args.n_points,
        val_per_class=args.val_per_class, train_per_class=args.train_per_class, test_per_class=args.test_per_class,
        cnn_epochs=args.cnn_epochs, cnn_batch_size=args.cnn_batch_size, cnn_learning_rate=args.cnn_learning_rate,
        resnet_head_epochs=args.resnet_head_epochs, resnet_patience=args.resnet_patience,
        resnet_batch_size=args.resnet_batch_size, resnet_feature_batch_size=args.resnet_feature_batch_size,
        resnet_learning_rate=args.resnet_learning_rate, eval_subset_size=args.eval_subset_size,
        output_root=args.output_root, quick=args.quick,
    )
    if cfg.quick:
        cfg.seeds = cfg.seeds[:2]
        cfg.n_points = min(cfg.n_points, 12)
        cfg.val_per_class = min(cfg.val_per_class, 50)
        cfg.train_per_class = 300 if cfg.train_per_class <= 0 else min(cfg.train_per_class, 300)
        cfg.test_per_class = 100 if cfg.test_per_class <= 0 else min(cfg.test_per_class, 100)
        cfg.cnn_epochs = min(cfg.cnn_epochs, 3)
        cfg.resnet_head_epochs = min(cfg.resnet_head_epochs, 3)
        cfg.eval_subset_size = min(cfg.eval_subset_size, 100) if cfg.eval_subset_size > 0 else 100
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    torch.manual_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def default_output_root() -> str:
    if os.path.exists("/kaggle/working"):
        return "/kaggle/working/srf_stability_outputs_en"
    return "srf_stability_experiment/outputs_en"


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
        ys.append(y[chosen].astype(np.int64))
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
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "n_train": len(x_train), "n_val": len(x_val), "n_test": len(x_test),
    }


def subsample_for_sweep(x: np.ndarray, y: np.ndarray, subset_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if subset_size is None or subset_size <= 0 or subset_size >= len(x):
        return x, y
    rng = np.random.default_rng(seed + 2000)
    idx = rng.choice(len(x), size=subset_size, replace=False)
    return x[idx], y[idx]


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


def build_frozen_resnet18(device: torch.device) -> nn.Module:
    try:
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    except AttributeError:
        backbone = torchvision.models.resnet18(pretrained=True)
    backbone.fc = nn.Identity()
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()
    backbone.to(device)
    return backbone


class LinearHead(nn.Module):
    def __init__(self, in_features: int = 512, n_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.fc(x)


def preprocess_for_resnet(images_hw: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(images_hw)).unsqueeze(1).to(device).float()
    x = Fnn.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return x


@torch.no_grad()
def extract_features(backbone: nn.Module, images_hw: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    feats = []
    for start in range(0, len(images_hw), batch_size):
        batch = images_hw[start:start + batch_size]
        x = preprocess_for_resnet(batch, device)
        feats.append(backbone(x).cpu().numpy())
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def evaluate_full_pipeline_torch(backbone: nn.Module, head: nn.Module, images_hw: np.ndarray, y: np.ndarray,
                                  device: torch.device, batch_size: int) -> float:
    correct = 0
    for start in range(0, len(images_hw), batch_size):
        batch = images_hw[start:start + batch_size]
        x = preprocess_for_resnet(batch, device)
        logits = head(backbone(x))
        preds = logits.argmax(dim=1).cpu().numpy()
        correct += int((preds == y[start:start + batch_size]).sum())
    return correct / len(images_hw)


def train_head(head: nn.Module, x_train_feat: np.ndarray, y_train: np.ndarray, x_val_feat: np.ndarray, y_val: np.ndarray,
               device: torch.device, epochs: int, batch_size: int, lr: float, patience: int) -> float:
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    x_train_t = torch.from_numpy(x_train_feat).float().to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    x_val_t = torch.from_numpy(x_val_feat).float().to(device)
    y_val_t = torch.from_numpy(y_val).long().to(device)

    best_val_acc, best_state, stale_epochs = -1.0, None, 0
    n = len(x_train_t)
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        running_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = Fnn.cross_entropy(head(x_train_t[idx]), y_train_t[idx])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(idx)
        head.eval()
        with torch.no_grad():
            val_acc = (head(x_val_t).argmax(dim=1) == y_val_t).float().mean().item()
        print(f"    epoch {epoch + 1}/{epochs} - loss={running_loss / n:.4f} - val_accuracy={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc, best_state, stale_epochs = val_acc, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    return best_val_acc


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


def sweep_cumulative_lowpass_cnn(model, x_test_img: np.ndarray, y_test: np.ndarray, height: int, width: int,
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
            "modes_kept": int(n_keep), "radius_cutoff": radius_cutoff,
            "normalized_radius": radius_cutoff / rho_max, "accuracy": float(accuracy),
        })
    return rows


def sweep_cumulative_lowpass_resnet(backbone: nn.Module, head: nn.Module, x_test_img: np.ndarray, y_test: np.ndarray,
                                     height: int, width: int, n_points: int, batch_size: int,
                                     device: torch.device) -> list[dict[str, float]]:
    radius = radial_frequency_grid_centered(height, width)
    order = np.argsort(radius, axis=None)
    sorted_radius = radius.flatten()[order]
    total_modes = height * width
    rho_max = float(sorted_radius[-1])
    counts = build_keep_count_schedule(total_modes, n_points)

    rows = []
    for n_keep in counts:
        degraded = reconstruct_with_keep_count(x_test_img, order, height, width, int(n_keep))
        accuracy = evaluate_full_pipeline_torch(backbone, head, degraded, y_test, device, batch_size)
        radius_cutoff = float(sorted_radius[n_keep - 1])
        rows.append({
            "modes_kept": int(n_keep), "radius_cutoff": radius_cutoff,
            "normalized_radius": radius_cutoff / rho_max, "accuracy": float(accuracy),
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


def fit_isotonic_cdf(normalized_accuracy: np.ndarray) -> np.ndarray:
    y = np.clip(np.asarray(normalized_accuracy, dtype=float), 0.0, 1.0)
    return np.clip(pool_adjacent_violators(y), 0.0, 1.0)


def cdf_statistics(normalized_radius: np.ndarray, cdf_values: np.ndarray) -> dict[str, Any]:
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

    widths_raw = np.diff(x)
    nonzero = widths_raw > 1e-12
    density_raw = np.zeros_like(widths_raw)
    density_raw[nonzero] = np.diff(f)[nonzero] / widths_raw[nonzero]
    mode_idx = int(np.argmax(density_raw))
    mode = float(0.5 * (x[mode_idx] + x[mode_idx + 1]))

    p = np.diff(f, prepend=0.0)
    p = np.clip(p, 0.0, None)
    eps = 1e-12
    entropy = float(-np.sum(np.where(p > eps, p * np.log(p), 0.0)))

    auc = float(_trapz(f, x))
    gaussian_entropy = float(0.5 * np.log(2 * np.pi * np.e * max(variance, eps)))
    entropy_gap = gaussian_entropy - entropy

    return {
        "mean": mean, "std": std, "median": median, "mode": mode, "auc": auc,
        "entropy": entropy, "entropy_gap": entropy_gap,
        "x": x.tolist(), "f": f.tolist(),
    }


def run_seed_cnn(dataset: str, cfg: ExperimentConfig, seed: int) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset]
    height, width = spec["image_shape"]

    print(f"  [CNN seed={seed}] training...")
    set_seed(seed)
    data = load_dataset_three_way(dataset, cfg, seed)
    cnn = build_cnn(cfg.cnn_learning_rate, (height, width))
    fit_with_early_stopping(
        cnn, data["x_train_img"][..., np.newaxis], data["y_train"],
        data["x_val_img"][..., np.newaxis], data["y_val"],
        cfg.cnn_epochs, cfg.cnn_batch_size, "val_accuracy",
    )
    _, baseline_acc = cnn.evaluate(data["x_test_img"][..., np.newaxis], data["y_test"], verbose=0)
    print(f"  [CNN seed={seed}] baseline accuracy = {baseline_acc*100:.2f}% ; sweeping spectrum...")

    rows = sweep_cumulative_lowpass_cnn(cnn, data["x_test_img"], data["y_test"], height, width, cfg.n_points, cfg.cnn_batch_size)
    accuracy_full = rows[-1]["accuracy"]
    for row in rows:
        row["normalized_accuracy"] = float(np.clip(row["accuracy"] / accuracy_full, 0.0, 1.0)) if accuracy_full > 0 else 0.0

    x = np.array([r["normalized_radius"] for r in rows])
    y = np.array([r["normalized_accuracy"] for r in rows])
    f_iso = fit_isotonic_cdf(y)
    stats = cdf_statistics(x, f_iso)
    return {"seed": seed, "baseline_accuracy": float(baseline_acc), "rows": rows, "stats": stats}


def run_seed_resnet(dataset: str, cfg: ExperimentConfig, seed: int, backbone: nn.Module, device: torch.device) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset]
    height, width = spec["image_shape"]

    print(f"  [ResNet-18 seed={seed}] extracting features and training the linear head...")
    set_seed(seed)
    data = load_dataset_three_way(dataset, cfg, seed)

    train_feats = extract_features(backbone, data["x_train_img"], device, cfg.resnet_feature_batch_size)
    val_feats = extract_features(backbone, data["x_val_img"], device, cfg.resnet_feature_batch_size)
    head = LinearHead(512, 10).to(device)
    val_acc = train_head(head, train_feats, data["y_train"], val_feats, data["y_val"], device,
                          cfg.resnet_head_epochs, cfg.resnet_batch_size, cfg.resnet_learning_rate, cfg.resnet_patience)

    baseline_acc = evaluate_full_pipeline_torch(backbone, head, data["x_test_img"], data["y_test"], device, cfg.resnet_batch_size)
    print(f"  [ResNet-18 seed={seed}] val_accuracy={val_acc*100:.2f}% ; baseline accuracy (full test set)={baseline_acc*100:.2f}% ; sweeping spectrum...")

    x_sweep, y_sweep = subsample_for_sweep(data["x_test_img"], data["y_test"], cfg.eval_subset_size, seed)
    rows = sweep_cumulative_lowpass_resnet(backbone, head, x_sweep, y_sweep, height, width, cfg.n_points, cfg.resnet_batch_size, device)
    accuracy_full = rows[-1]["accuracy"]
    for row in rows:
        row["normalized_accuracy"] = float(np.clip(row["accuracy"] / accuracy_full, 0.0, 1.0)) if accuracy_full > 0 else 0.0

    x = np.array([r["normalized_radius"] for r in rows])
    y = np.array([r["normalized_accuracy"] for r in rows])
    f_iso = fit_isotonic_cdf(y)
    stats = cdf_statistics(x, f_iso)
    return {"seed": seed, "baseline_accuracy": float(baseline_acc), "rows": rows, "stats": stats}


def renormalize_srf(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y0 = y[0]
    denom = max(1.0 - y0, 1e-12)
    return np.clip((y - y0) / denom, 0.0, 1.0)


def aggregate_architecture_results(dataset: str, architecture: str, seeds: list[int], per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    x_common = np.array(per_seed[0]["stats"]["x"])
    f_matrix = np.array([renormalize_srf(np.array(s["stats"]["f"])) for s in per_seed])
    f_mean = f_matrix.mean(axis=0)
    f_std = f_matrix.std(axis=0)

    agg_stats: dict[str, dict[str, float]] = {}
    for key in ("mean", "std", "median", "mode", "auc", "entropy", "entropy_gap"):
        values = [s["stats"][key] for s in per_seed]
        agg_stats[key] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}
    baseline_accs = [s["baseline_accuracy"] for s in per_seed]
    agg_stats["baseline_accuracy"] = {"mean": float(np.mean(baseline_accs)), "std": float(np.std(baseline_accs)), "values": baseline_accs}

    return {
        "dataset": dataset, "display_name": DATASET_SPECS[dataset]["display_name"],
        "architecture": architecture, "architecture_display": ARCH_DISPLAY[architecture],
        "seeds": seeds, "per_seed": per_seed,
        "x_common": x_common.tolist(), "f_mean": f_mean.tolist(), "f_std": f_std.tolist(),
        "aggregate_stats": agg_stats,
    }


def save_stability_panel(ax, result: dict[str, Any]) -> None:
    x_common = np.array(result["x_common"])
    f_mean = np.array(result["f_mean"])
    f_std = np.array(result["f_std"])

    for idx, entry in enumerate(result["per_seed"]):
        x = entry["stats"]["x"]
        f_srf = renormalize_srf(np.array(entry["stats"]["f"]))
        ax.plot(x, f_srf, color="C0", alpha=0.25, linewidth=1.0,
                label=f"individual curves ({len(result['seeds'])} seeds)" if idx == 0 else None)

    ax.fill_between(x_common, np.clip(f_mean - f_std, 0, 1), np.clip(f_mean + f_std, 0, 1),
                     color="C1", alpha=0.25, label="± 1 standard deviation band")
    ax.plot(x_common, f_mean, color="C1", linewidth=2.5, label="mean curve")

    ax.set_xlabel(f"Normalized spectral radius (r = ρ/ρ_max) — {result['architecture_display']}")
    ax.set_ylabel("SRF(r)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)


def save_combined_plot(results: list[dict[str, Any]], dataset_display: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(6.5 * len(results), 5.5), squeeze=False)
    for ax, result in zip(axes[0], results):
        save_stability_panel(ax, result)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_report(output_dir: Path, cfg: ExperimentConfig, results: list[dict[str, Any]]) -> None:
    dataset_display = DATASET_SPECS[cfg.dataset]["display_name"]
    lines = [
        f"# SRF stability across seeds — {dataset_display}",
        "",
        "## Setup",
        f"- Dataset: `{dataset_display}`",
        f"- Architectures: `{cfg.architectures}`",
        f"- Seeds: `{cfg.seeds}` ({len(cfg.seeds)} independent training runs per architecture)",
        f"- Points per curve: `{cfg.n_points}`",
        "",
    ]
    labels = {
        "mean": "E[R]", "std": "σ[R]", "median": "Median", "mode": "Mode",
        "auc": "AUC", "entropy": "H", "entropy_gap": "ΔH (negentropy)",
    }

    lines += ["## Comparison across architectures (mean ± standard deviation across seeds)", "",
              "| Architecture | Baseline accuracy | E[R] | σ[R] | Median | Mode | AUC | H | ΔH |",
              "|---|---|---|---|---|---|---|---|---|"]
    for result in results:
        agg = result["aggregate_stats"]
        lines.append(
            f"| {result['architecture_display']} "
            f"| {agg['baseline_accuracy']['mean']*100:.2f}% ± {agg['baseline_accuracy']['std']*100:.2f}pp "
            f"| {agg['mean']['mean']:.4f} ± {agg['mean']['std']:.4f} "
            f"| {agg['std']['mean']:.4f} ± {agg['std']['std']:.4f} "
            f"| {agg['median']['mean']:.4f} ± {agg['median']['std']:.4f} "
            f"| {agg['mode']['mean']:.4f} ± {agg['mode']['std']:.4f} "
            f"| {agg['auc']['mean']:.4f} ± {agg['auc']['std']:.4f} "
            f"| {agg['entropy']['mean']:.4f} ± {agg['entropy']['std']:.4f} "
            f"| {agg['entropy_gap']['mean']:.4f} ± {agg['entropy_gap']['std']:.4f} |"
        )

    for result in results:
        agg = result["aggregate_stats"]
        lines += [
            "",
            f"## Per-seed detail — {result['architecture_display']}",
            "",
            "| Statistic | Mean | Std. dev. | Coefficient of variation (|σ/mean|) |",
            "|---|---|---|---|",
        ]
        for key, label in labels.items():
            m, s = agg[key]["mean"], agg[key]["std"]
            cv = abs(s / m) if abs(m) > 1e-9 else float("nan")
            lines.append(f"| {label} | {m:.4f} | {s:.4f} | {cv:.3f} |")
        lines += [
            "",
            "| Seed | Baseline accuracy | E[R] | σ[R] | Median | Mode | AUC | H | ΔH |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for entry in result["per_seed"]:
            s = entry["stats"]
            lines.append(
                f"| {entry['seed']} | {entry['baseline_accuracy']*100:.2f}% "
                f"| {s['mean']:.4f} | {s['std']:.4f} | {s['median']:.4f} | {s['mode']:.4f} "
                f"| {s['auc']:.4f} | {s['entropy']:.4f} | {s['entropy_gap']:.4f} |"
            )

    lines += [
        "",
        "A low coefficient of variation (small standard deviation relative to "
        "the mean) across all statistics indicates that the SRF is a "
        "reproducible characteristic of the architecture trained under this "
        "fixed configuration, and not an artifact of a particular "
        "initialization. Comparing the two architectures also makes it "
        "possible to check whether that reproducibility differs between a "
        "network trained from scratch and a pretrained network with a "
        "frozen backbone.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def resolve_torch_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            _ = torch.zeros(1, device=device) + 1.0
        except RuntimeError as exc:
            print(f"WARNING: a GPU was detected but it is not compatible with this PyTorch build ({exc}). "
                  f"On Kaggle this happens with the 'GPU P100' accelerator (Pascal architecture, sm_60): the "
                  f"preinstalled PyTorch only supports sm_70 and above. Switch the notebook accelerator "
                  f"to 'GPU T4 x2' (Settings -> Accelerator) and re-run. Falling back to CPU for now "
                  f"(this will be very slow; use --quick for a quick test).")
            device = torch.device("cpu")
    if device.type == "cpu":
        print("WARNING: running the ResNet-18 on CPU. At 224x224 with many sweep points per "
              "seed this will be very slow; use --quick for a quick test or run in an environment with a compatible GPU.")
    return device


def main() -> None:
    cfg = parse_args()
    output_dir = make_output_dir(cfg.output_root)
    dataset_display = DATASET_SPECS[cfg.dataset]["display_name"]
    print(f"Dataset: {dataset_display}  |  Architectures: {cfg.architectures}  |  Seeds: {cfg.seeds}")
    print(f"Output directory: {output_dir.resolve()}")

    results = []

    if "cnn" in cfg.architectures:
        print(f"\n===== ARCHITECTURE: {ARCH_DISPLAY['cnn']} =====")
        per_seed_cnn = [run_seed_cnn(cfg.dataset, cfg, seed) for seed in cfg.seeds]
        result_cnn = aggregate_architecture_results(cfg.dataset, "cnn", cfg.seeds, per_seed_cnn)
        results.append(result_cnn)
        with (output_dir / "cnn_stability_metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(result_cnn, fh, indent=2)

    if "resnet18" in cfg.architectures:
        print(f"\n===== ARCHITECTURE: {ARCH_DISPLAY['resnet18']} =====")
        device = resolve_torch_device()
        print(f"Device (PyTorch): {device}")
        backbone = build_frozen_resnet18(device)
        per_seed_resnet = [run_seed_resnet(cfg.dataset, cfg, seed, backbone, device) for seed in cfg.seeds]
        result_resnet = aggregate_architecture_results(cfg.dataset, "resnet18", cfg.seeds, per_seed_resnet)
        results.append(result_resnet)
        with (output_dir / "resnet18_stability_metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(result_resnet, fh, indent=2)

    save_combined_plot(results, dataset_display, output_dir / f"{cfg.dataset}_srf_stability_comparison.png")
    build_report(output_dir, cfg, results)

    with (output_dir / "metrics_all_architectures.json").open("w", encoding="utf-8") as fh:
        json.dump({"config": asdict(cfg), "results": results}, fh, indent=2)

    print("\n=== Aggregate summary ===")
    for result in results:
        agg = result["aggregate_stats"]
        print(
            f"{result['architecture_display']} ({dataset_display}, {len(cfg.seeds)} seeds): "
            f"baseline_accuracy={agg['baseline_accuracy']['mean']*100:.2f}%±{agg['baseline_accuracy']['std']*100:.2f}pp "
            f"E[R]={agg['mean']['mean']:.4f}±{agg['mean']['std']:.4f} "
            f"σ[R]={agg['std']['mean']:.4f}±{agg['std']['std']:.4f} "
            f"median={agg['median']['mean']:.4f}±{agg['median']['std']:.4f} "
            f"mode={agg['mode']['mean']:.4f}±{agg['mode']['std']:.4f} "
            f"AUC={agg['auc']['mean']:.4f}±{agg['auc']['std']:.4f} "
            f"H={agg['entropy']['mean']:.4f}±{agg['entropy']['std']:.4f} "
            f"ΔH={agg['entropy_gap']['mean']:.4f}±{agg['entropy_gap']['std']:.4f}"
        )
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
