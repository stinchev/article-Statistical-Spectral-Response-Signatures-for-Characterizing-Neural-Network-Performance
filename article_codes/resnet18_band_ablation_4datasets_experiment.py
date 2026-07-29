from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import torchvision


MODE_LABEL_EN = {"high": "high frequencies", "low": "low frequencies", "mid": "mid frequencies"}
BAND_LABEL_EN = {"low": "low band", "mid": "mid band", "high": "high band"}

DATASET_SPECS = {
    "mnist": {"display_name": "MNIST", "image_shape": (28, 28), "torch_cls": torchvision.datasets.MNIST,
              "band_low_max": 9.2195, "band_high_min": 13.0},
    "fashion_mnist": {"display_name": "Fashion-MNIST", "image_shape": (28, 28), "torch_cls": torchvision.datasets.FashionMNIST,
                       "band_low_max": 9.2195, "band_high_min": 13.0},
    "kmnist": {"display_name": "KMNIST", "image_shape": (28, 28), "torch_cls": torchvision.datasets.KMNIST,
               "band_low_max": 9.2195, "band_high_min": 13.0},
    "cifar10": {"display_name": "CIFAR-10 (grayscale)", "image_shape": (32, 32), "torch_cls": torchvision.datasets.CIFAR10,
                "band_low_max": 10.3, "band_high_min": 14.8},
}
DATASET_ORDER = ["mnist", "fashion_mnist", "kmnist", "cifar10"]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _robust_torchvision_cifar10_download(data_root, max_retries=10):
    import hashlib
    import time
    import urllib.request

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    expected_md5 = "c58f30108f718f92721af3b95e74349a"
    os.makedirs(data_root, exist_ok=True)
    target = os.path.join(data_root, "cifar-10-python.tar.gz")

    def md5_of(path):
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    if os.path.exists(target) and md5_of(target) == expected_md5:
        return

    for attempt in range(1, max_retries + 1):
        try:
            resume_from = os.path.getsize(target) if os.path.exists(target) else 0
            req = urllib.request.Request(url)
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
            if md5_of(target) == expected_md5:
                return
            print(f"CIFAR-10 archive md5 mismatch after attempt {attempt}/{max_retries}, retrying...")
        except Exception as exc:
            print(f"CIFAR-10 download attempt {attempt}/{max_retries} failed ({exc}); retrying in 5s...")
            time.sleep(5)
    raise RuntimeError(f"Could not download a valid CIFAR-10 archive after {max_retries} attempts.")


def load_dataset_raw(dataset: str, data_root: str):
    spec = DATASET_SPECS[dataset]
    cls = spec["torch_cls"]
    if dataset == "cifar10":
        _robust_torchvision_cifar10_download(data_root)
    train_ds = cls(root=data_root, train=True, download=True)
    test_ds = cls(root=data_root, train=False, download=True)
    if dataset == "cifar10":
        x_train = train_ds.data.astype("float32")
        x_test = test_ds.data.astype("float32")
        y_train = np.array(train_ds.targets, dtype=np.int64)
        y_test = np.array(test_ds.targets, dtype=np.int64)

        def to_gray(x):
            return 0.2989 * x[..., 0] + 0.5870 * x[..., 1] + 0.1140 * x[..., 2]

        x_train, x_test = to_gray(x_train), to_gray(x_test)
    else:
        x_train = train_ds.data.numpy().astype("float32")
        x_test = test_ds.data.numpy().astype("float32")
        y_train = train_ds.targets.numpy().astype(np.int64)
        y_test = test_ds.targets.numpy().astype(np.int64)
    return (x_train, y_train), (x_test, y_test)


@dataclass
class ExperimentConfig:
    datasets: list[str] = field(default_factory=lambda: list(DATASET_ORDER))
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    eval_subset_size: int = 2000
    val_per_class: int = 500
    train_per_class: int = -1
    test_per_class: int = -1
    head_epochs: int = 30
    patience: int = 5
    batch_size: int = 128
    feature_batch_size: int = 256
    learning_rate: float = 1e-3
    output_root: str = ""
    data_root: str = ""
    ablation_step: float = 0.05
    quick: bool = False


def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Frozen ResNet-18 band ablation (progressive sweep + complete removal), 4 datasets.")
    parser.add_argument("--datasets", type=str, nargs="+", default=list(DATASET_ORDER), choices=sorted(DATASET_SPECS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--eval-subset-size", type=int, default=2000, help="-1 to use the full test set in the progressive sweep")
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--train-per-class", type=int, default=-1)
    parser.add_argument("--test-per-class", type=int, default=-1)
    parser.add_argument("--head-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--ablation-step", type=float, default=0.05)
    parser.add_argument("--quick", action="store_true")
    args, unknown_args = parser.parse_known_args(argv)
    if unknown_args:
        print(f"Ignoring notebook/kernel arguments: {unknown_args}")

    cfg = ExperimentConfig(
        datasets=args.datasets, seeds=args.seeds, eval_subset_size=args.eval_subset_size,
        val_per_class=args.val_per_class, train_per_class=args.train_per_class, test_per_class=args.test_per_class,
        head_epochs=args.head_epochs, patience=args.patience, batch_size=args.batch_size,
        feature_batch_size=args.feature_batch_size, learning_rate=args.learning_rate,
        output_root=args.output_root, data_root=args.data_root, ablation_step=args.ablation_step, quick=args.quick,
    )
    if cfg.quick:
        cfg.seeds = cfg.seeds[:1]
        cfg.eval_subset_size = min(cfg.eval_subset_size, 100) if cfg.eval_subset_size > 0 else 100
        cfg.val_per_class = min(cfg.val_per_class, 30)
        cfg.train_per_class = 200 if cfg.train_per_class <= 0 else min(cfg.train_per_class, 200)
        cfg.test_per_class = 50 if cfg.test_per_class <= 0 else min(cfg.test_per_class, 50)
        cfg.head_epochs = min(cfg.head_epochs, 3)
        cfg.ablation_step = max(cfg.ablation_step, 0.25)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_output_root() -> str:
    if os.path.exists("/kaggle/working"):
        return "/kaggle/working/resnet18_band_ablation_outputs_en"
    return "resnet18_band_ablation_experiment/outputs_en"


def default_data_root() -> str:
    if os.path.exists("/kaggle/working"):
        return "/kaggle/working/torch_data"
    return "resnet18_band_ablation_experiment/torch_data"


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
    data_root = cfg.data_root or default_data_root()
    (x_train_full, y_train_full), (x_test_full, y_test_full) = load_dataset_raw(dataset, data_root)
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
def evaluate_full_pipeline(backbone: nn.Module, head: nn.Module, images_hw: np.ndarray, y: np.ndarray,
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


def apply_mask(images: np.ndarray, mask: np.ndarray) -> np.ndarray:
    shifted = np.fft.fftshift(np.fft.fft2(images, axes=(-2, -1)), axes=(-2, -1))
    restored = np.fft.ifft2(np.fft.ifftshift(shifted * mask[np.newaxis], axes=(-2, -1)), axes=(-2, -1)).real
    return np.clip(restored, 0.0, 1.0)


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


def mask_energy_stats(images: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    total = int(mask.size)
    kept = int(mask.sum())
    return {"kept_coefficients": kept, "removed_coefficients": total - kept}


def evaluate_band_ablation(backbone, head, x_test_img, y_test, band_low_max, band_high_min, device, batch_size) -> list[dict]:
    baseline_acc = evaluate_full_pipeline(backbone, head, x_test_img, y_test, device, batch_size)
    rows = []
    for band in ("low", "mid", "high"):
        mask = band_keep_mask(x_test_img.shape[1], x_test_img.shape[2], band, band_low_max, band_high_min)
        degraded = apply_mask(x_test_img, mask)
        accuracy = evaluate_full_pipeline(backbone, head, degraded, y_test, device, batch_size)
        row = {"band": band, "accuracy": float(accuracy)}
        row.update(mask_energy_stats(x_test_img, mask))
        row["accuracy_drop_from_baseline_pp"] = 100.0 * (baseline_acc - accuracy)
        rows.append(row)
    return rows


def evaluate_ablation(backbone, head, x_test_img, y_test, ablation_step, mode, band_mid_rho, device, batch_size):
    rows = []
    for remove_fraction in np.arange(0.0, 0.951, ablation_step):
        remove_fraction = float(remove_fraction)
        h, w = x_test_img.shape[1], x_test_img.shape[2]
        if mode == "high":
            mask = radial_keep_mask(h, w, remove_fraction)
        elif mode == "low":
            mask = low_frequency_keep_mask(h, w, remove_fraction)
        else:
            mask = mid_frequency_keep_mask(h, w, remove_fraction, band_mid_rho)
        degraded = apply_mask(x_test_img, mask)
        accuracy = evaluate_full_pipeline(backbone, head, degraded, y_test, device, batch_size)
        row = {"removed_fraction_key": remove_fraction, "mode": mode, "accuracy": float(accuracy)}
        row.update(mask_energy_stats(x_test_img, mask))
        rows.append(row)
    baseline = rows[0]["accuracy"]
    prev = baseline
    for idx, row in enumerate(rows):
        row["accuracy_drop_from_baseline_pp"] = 100.0 * (baseline - row["accuracy"])
        row["incremental_accuracy_drop_pp"] = 0.0 if idx == 0 else 100.0 * (prev - row["accuracy"])
        prev = row["accuracy"]
    return rows


def run_dataset(dataset: str, cfg: ExperimentConfig, backbone: nn.Module, device: torch.device) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset]
    height, width = spec["image_shape"]
    band_low_max, band_high_min = spec["band_low_max"], spec["band_high_min"]
    band_mid_rho = (band_low_max + band_high_min) / 2.0

    per_seed_results = []
    for seed in cfg.seeds:
        print(f"  [seed={seed}] extracting features and training the linear head...")
        set_seed(seed)
        data = load_dataset_three_way(dataset, cfg, seed)

        train_feats = extract_features(backbone, data["x_train_img"], device, cfg.feature_batch_size)
        val_feats = extract_features(backbone, data["x_val_img"], device, cfg.feature_batch_size)
        head = LinearHead(512, 10).to(device)
        val_acc = train_head(head, train_feats, data["y_train"], val_feats, data["y_val"], device,
                              cfg.head_epochs, cfg.batch_size, cfg.learning_rate, cfg.patience)

        baseline_acc = evaluate_full_pipeline(backbone, head, data["x_test_img"], data["y_test"], device, cfg.batch_size)
        print(f"  [seed={seed}] val_accuracy={val_acc*100:.2f}% ; baseline accuracy (full test set)={baseline_acc*100:.2f}%")

        band_rows = evaluate_band_ablation(backbone, head, data["x_test_img"], data["y_test"],
                                            band_low_max, band_high_min, device, cfg.batch_size)

        x_sweep, y_sweep = subsample_for_sweep(data["x_test_img"], data["y_test"], cfg.eval_subset_size, seed)
        high_rows = evaluate_ablation(backbone, head, x_sweep, y_sweep, cfg.ablation_step, "high", band_mid_rho, device, cfg.batch_size)
        low_rows = evaluate_ablation(backbone, head, x_sweep, y_sweep, cfg.ablation_step, "low", band_mid_rho, device, cfg.batch_size)
        mid_rows = evaluate_ablation(backbone, head, x_sweep, y_sweep, cfg.ablation_step, "mid", band_mid_rho, device, cfg.batch_size)

        per_seed_results.append({
            "seed": seed, "n_train": data["n_train"], "n_val": data["n_val"], "n_test": data["n_test"],
            "test_accuracy": float(baseline_acc),
            "high_frequency_ablation": high_rows, "low_frequency_ablation": low_rows,
            "mid_frequency_ablation": mid_rows, "band_ablation": band_rows,
        })

    agg = aggregate_seeds(per_seed_results)
    return {
        "dataset": dataset, "display_name": spec["display_name"], "image_shape": [height, width],
        "eval_subset_size": cfg.eval_subset_size,
        "per_seed_results": per_seed_results, "aggregate": agg,
    }


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "values": [float(v) for v in arr]}


def aggregate_seeds(per_seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, Any] = {"n_seeds": len(per_seed_results), "seeds": [r["seed"] for r in per_seed_results]}
    agg["test_accuracy"] = mean_std([r["test_accuracy"] for r in per_seed_results])
    agg["band_ablation"] = {}
    for band_idx, band in enumerate(("low", "mid", "high")):
        agg["band_ablation"][band] = {
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
    plt.xlabel(f"{MODE_LABEL_EN[mode].capitalize()} coefficients removed (%) — Frozen ResNet-18, {display_name}")
    plt.ylabel("Test accuracy (%)")
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
    plt.ylabel("Test accuracy (%)")
    plt.xlabel(f"Frozen ResNet-18 — {display_name}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def build_report(output_dir: Path, cfg: ExperimentConfig, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Frozen ResNet-18 band ablation (progressive sweep + complete removal), 4 datasets",
        "",
        "## Setup",
        f"- Seeds: `{cfg.seeds}`",
        f"- Progressive sweep step: `{cfg.ablation_step*100:.0f}%`",
        f"- Test subset for the progressive sweep: `{cfg.eval_subset_size if cfg.eval_subset_size > 0 else 'full'}`",
        "",
    ]
    for result in results:
        agg = result["aggregate"]
        lines += [
            f"## {result['display_name']}",
            f"- Baseline accuracy (full test set): `{agg['test_accuracy']['mean']*100:.2f}% ± {agg['test_accuracy']['std']*100:.2f}pp`",
            "",
            "| Band removed | Modes removed | Accuracy | Drop |",
            "|---|---|---|---|",
        ]
        for band in ("low", "mid", "high"):
            band_res = agg["band_ablation"][band]
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            _ = torch.zeros(1, device=device) + 1.0
        except RuntimeError as exc:
            print(f"WARNING: a GPU was detected but it is not compatible with this PyTorch build ({exc}). "
                  f"On Kaggle this happens with the 'GPU P100' accelerator (Pascal architecture, sm_60): the "
                  f"preinstalled PyTorch only supports sm_70 and above. Switch the notebook accelerator "
                  f"to 'GPU T4 x2' (Settings -> Accelerator) and re-run. Falling back to CPU for now.")
            device = torch.device("cpu")
    if device.type == "cpu":
        print("WARNING: running on CPU. With ResNet-18 at 224x224 and several datasets this will be very slow; use --quick.")
    print(f"Device: {device}")
    print(f"Datasets: {cfg.datasets}")
    print(f"Output directory: {output_dir.resolve()}")

    backbone = build_frozen_resnet18(device)

    results = []
    for dataset in cfg.datasets:
        print(f"\n===== DATASET: {DATASET_SPECS[dataset]['display_name']} =====")
        result = run_dataset(dataset, cfg, backbone, device)
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
        print(f"{result['display_name']}: baseline_accuracy={agg['test_accuracy']['mean']*100:.2f}%±{agg['test_accuracy']['std']*100:.2f}pp")
        for band in ("low", "mid", "high"):
            band_res = agg["band_ablation"][band]
            print(f"  {band} band: accuracy={band_res['accuracy']['mean']*100:.2f}%±{band_res['accuracy']['std']*100:.2f}pp "
                  f"drop={band_res['accuracy_drop_from_baseline_pp']['mean']:.2f}pp")
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
