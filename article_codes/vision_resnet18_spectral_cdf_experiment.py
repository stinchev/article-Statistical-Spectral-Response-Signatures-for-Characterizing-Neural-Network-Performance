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


_trapz = getattr(np, "trapezoid", None) or np.trapz


DATASET_SPECS = {
    "mnist": {"display_name": "MNIST", "image_shape": (28, 28), "torch_cls": torchvision.datasets.MNIST},
    "fashion_mnist": {"display_name": "Fashion-MNIST", "image_shape": (28, 28), "torch_cls": torchvision.datasets.FashionMNIST},
    "kmnist": {"display_name": "KMNIST", "image_shape": (28, 28), "torch_cls": torchvision.datasets.KMNIST},
    "cifar10": {"display_name": "CIFAR-10 (grayscale)", "image_shape": (32, 32), "torch_cls": torchvision.datasets.CIFAR10},
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


def load_dataset_raw(dataset: str, data_root: str) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
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
    n_points: int = 150
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
    quick: bool = False


def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Cumulative spectral sweep + CDF with pretrained ResNet-18 (frozen backbone).")
    parser.add_argument("--datasets", type=str, nargs="+", default=list(DATASET_ORDER), choices=sorted(DATASET_SPECS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--n-points", type=int, default=150)
    parser.add_argument("--eval-subset-size", type=int, default=2000, help="-1 to use the full test set in the sweep")
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
    parser.add_argument("--quick", action="store_true")
    args, unknown_args = parser.parse_known_args(argv)
    if unknown_args:
        print(f"Ignoring notebook/kernel arguments: {unknown_args}")

    cfg = ExperimentConfig(
        datasets=args.datasets, seeds=args.seeds, n_points=args.n_points, eval_subset_size=args.eval_subset_size,
        val_per_class=args.val_per_class, train_per_class=args.train_per_class, test_per_class=args.test_per_class,
        head_epochs=args.head_epochs, patience=args.patience, batch_size=args.batch_size,
        feature_batch_size=args.feature_batch_size, learning_rate=args.learning_rate,
        output_root=args.output_root, data_root=args.data_root, quick=args.quick,
    )
    if cfg.quick:
        cfg.seeds = cfg.seeds[:1]
        cfg.n_points = min(cfg.n_points, 12)
        cfg.eval_subset_size = min(cfg.eval_subset_size, 100) if cfg.eval_subset_size > 0 else 100
        cfg.val_per_class = min(cfg.val_per_class, 30)
        cfg.train_per_class = 200 if cfg.train_per_class <= 0 else min(cfg.train_per_class, 200)
        cfg.test_per_class = 50 if cfg.test_per_class <= 0 else min(cfg.test_per_class, 50)
        cfg.head_epochs = min(cfg.head_epochs, 3)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_output_root() -> str:
    if os.path.exists("/kaggle/working"):
        return "/kaggle/working/resnet18_spectral_cdf_outputs_en"
    return "resnet18_spectral_cdf_experiment/outputs_en"


def default_data_root() -> str:
    if os.path.exists("/kaggle/working"):
        return "/kaggle/working/torch_data"
    return "resnet18_spectral_cdf_experiment/torch_data"


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


def sweep_cumulative_lowpass(backbone: nn.Module, head: nn.Module, x_test_img: np.ndarray, y_test: np.ndarray,
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
        accuracy = evaluate_full_pipeline(backbone, head, degraded, y_test, device, batch_size)
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


def fit_isotonic_cdf(normalized_accuracy: np.ndarray) -> np.ndarray:
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


def run_dataset(dataset: str, cfg: ExperimentConfig, backbone: nn.Module, device: torch.device) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset]
    height, width = spec["image_shape"]
    per_seed_rows, per_seed_iso, per_seed_stats, baseline_accs = [], [], [], []

    for seed in cfg.seeds:
        print(f"  [seed={seed}] extracting frozen features and training the linear head...")
        set_seed(seed)
        data = load_dataset_three_way(dataset, cfg, seed)

        train_feats = extract_features(backbone, data["x_train_img"], device, cfg.feature_batch_size)
        val_feats = extract_features(backbone, data["x_val_img"], device, cfg.feature_batch_size)
        head = LinearHead(512, 10).to(device)
        val_acc = train_head(head, train_feats, data["y_train"], val_feats, data["y_val"], device,
                              cfg.head_epochs, cfg.batch_size, cfg.learning_rate, cfg.patience)

        baseline_acc = evaluate_full_pipeline(backbone, head, data["x_test_img"], data["y_test"], device, cfg.batch_size)
        print(f"  [seed={seed}] val_accuracy={val_acc*100:.2f}% ; baseline accuracy (full test set)={baseline_acc*100:.2f}% ; sweeping spectrum...")

        x_sweep, y_sweep = subsample_for_sweep(data["x_test_img"], data["y_test"], cfg.eval_subset_size, seed)
        rows = sweep_cumulative_lowpass(backbone, head, x_sweep, y_sweep, height, width, cfg.n_points, cfg.batch_size, device)
        accuracy_full = rows[-1]["accuracy"]
        for row in rows:
            row["normalized_accuracy"] = float(np.clip(row["accuracy"] / accuracy_full, 0.0, 1.0)) if accuracy_full > 0 else 0.0

        x = np.array([r["normalized_radius"] for r in rows])
        y = np.array([r["normalized_accuracy"] for r in rows])
        f_iso = fit_isotonic_cdf(y)
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
        "eval_subset_size": cfg.eval_subset_size,
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
        plt.plot(x, y_srf, color=f"C{seed_idx}", label=f"seed {seed_idx} — frozen ResNet-18")
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
    plt.xlabel("Normalized spectral radius (r = ρ/ρ_max) — frozen ResNet-18")
    plt.ylabel("Mean SRF(r)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def build_report(output_dir: Path, cfg: ExperimentConfig, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Cumulative spectral sweep + CDF — pretrained ResNet-18 (frozen backbone, linear head)",
        "",
        "## Setup",
        f"- Seeds: `{cfg.seeds}`",
        f"- Points per curve: `{cfg.n_points}`",
        f"- Size of the test subset used in the sweep: `{cfg.eval_subset_size if cfg.eval_subset_size > 0 else 'full'}`",
        "",
        "## Statistics of the implicit distribution (normalized spectral radius, mean ± std across seeds)",
        "",
        "| Dataset | Baseline accuracy (full test set) | E[r] (mean) | Std. dev. | Median | Mode |",
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
        "after fitting it to a monotone curve via isotonic regression. ResNet-18 "
        "backbone pretrained on ImageNet and frozen; only the linear head is trained.",
        "",
    ]
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
                  f"to 'GPU T4 x2' (Settings -> Accelerator) and re-run. Falling back to CPU for now "
                  f"(this will be very slow; use --quick for a quick test).")
            device = torch.device("cpu")
    if device.type == "cpu":
        print("WARNING: running on CPU. ResNet-18 at 224x224 with ~150 sweep points per seed will be "
              "very slow; use --quick for a quick test or run this in an environment with a compatible GPU.")
    print(f"Device: {device}")
    print(f"Datasets: {cfg.datasets}")
    print(f"Output directory: {output_dir.resolve()}")

    backbone = build_frozen_resnet18(device)

    results = []
    for dataset in cfg.datasets:
        print(f"\n===== DATASET: {DATASET_SPECS[dataset]['display_name']} =====")
        result = run_dataset(dataset, cfg, backbone, device)
        results.append(result)
        save_dataset_curve_plot(result, output_dir / f"{dataset}_resnet18_spectral_cdf.png")

        with (output_dir / f"{dataset}_metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

    save_comparison_plot(results, output_dir / "comparison_resnet18_spectral_cdf.png")
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
