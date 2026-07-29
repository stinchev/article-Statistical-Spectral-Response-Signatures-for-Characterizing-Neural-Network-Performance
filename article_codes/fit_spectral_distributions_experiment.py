import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import minimize

_trapz = getattr(np, "trapezoid", None) or np.trapz


LOCAL_NETWORKS = {
    "Small CNN": Path(r"C:\Users\stst1\Downloads\results (52)\spectral_cdf_outputs\20260728_171111"),
    "Frozen ResNet-18": Path(r"C:\Users\stst1\Downloads\results (53)\resnet18_spectral_cdf_outputs\20260728_171133"),
}
LOCAL_OUTPUT_DIR = Path(r"C:\Users\stst1\Downloads\spectral_distribution_fits_en")

DATASETS = ["mnist", "fashion_mnist", "kmnist", "cifar10"]
DISPLAY = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST", "kmnist": "KMNIST", "cifar10": "CIFAR-10 (grayscale)"}


def find_output_dir(roots: list[Path], folder_names: list[str]) -> Path:
    for folder_name in folder_names:
        for root in roots:
            if not root.exists():
                continue
            matches = [p for p in root.rglob(folder_name) if p.is_dir()]
            if matches:
                base = matches[0]
                subdirs = [d for d in base.iterdir() if d.is_dir()]
                return sorted(subdirs, key=lambda d: d.name)[-1] if subdirs else base
    raise FileNotFoundError(
        f"Could not find any of {folder_names} under {[str(r) for r in roots]}. "
        f"Either run the corresponding sweep script earlier in this same Kaggle session, or "
        f"attach its notebook output as a data source (Add Data -> Your Work -> pick the "
        f"notebook version with the results) and re-run."
    )


def default_networks() -> dict[str, Path]:
    kaggle_roots = [Path("/kaggle/working"), Path("/kaggle/input")]
    if any(root.exists() for root in kaggle_roots):
        try:
            return {
                "Small CNN": find_output_dir(kaggle_roots, ["spectral_cdf_outputs_en", "spectral_cdf_outputs"]),
                "Frozen ResNet-18": find_output_dir(
                    kaggle_roots, ["resnet18_spectral_cdf_outputs_en", "resnet18_spectral_cdf_outputs"]
                ),
            }
        except FileNotFoundError as exc:
            print(f"{exc}\nFalling back to LOCAL_NETWORKS.")
    return LOCAL_NETWORKS


def default_output_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/spectral_distribution_fits_en")
    return LOCAL_OUTPUT_DIR


NETWORKS = default_networks()
OUTPUT_DIR = default_output_dir()
print(f"Networks detected: {[(name, str(path)) for name, path in NETWORKS.items()]}")
print(f"Output directory: {OUTPUT_DIR}")


def renormalize_srf(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y0 = y[0]
    denom = max(1.0 - y0, 1e-12)
    return np.clip((y - y0) / denom, 0.0, 1.0)


def load_empirical_cdf(base_dir: Path, dataset: str):
    with (base_dir / f"{dataset}_metrics.json").open(encoding="utf-8") as fh:
        result = json.load(fh)
    rows0 = result["per_seed_rows"][0]
    x = np.array([r["normalized_radius"] for r in rows0], dtype=float)
    iso_curves = np.array(result["per_seed_isotonic_cdf"], dtype=float)
    srf_curves = np.array([renormalize_srf(c) for c in iso_curves])
    f_mean = srf_curves.mean(axis=0)


    if x[0] > 0.0:
        x = np.concatenate([[0.0], x])
        f_mean = np.concatenate([[0.0], f_mean])
    if x[-1] < 1.0:
        x = np.concatenate([x, [1.0]])
        f_mean = np.concatenate([f_mean, [1.0]])
    return x, f_mean


def truncated_cdf(base_cdf_fn, x, params):
    f0 = base_cdf_fn(x, *params)
    f0_0 = base_cdf_fn(np.array([0.0]), *params)[0]
    f0_1 = base_cdf_fn(np.array([1.0]), *params)[0]
    denom = max(f0_1 - f0_0, 1e-12)
    return np.clip((f0 - f0_0) / denom, 0.0, 1.0)


def beta_moment_init(mean, std):
    mean = min(max(mean, 1e-3), 1 - 1e-3)
    var = min(std ** 2, mean * (1 - mean) * 0.98)
    common = mean * (1 - mean) / var - 1
    a = max(mean * common, 1e-2)
    b = max((1 - mean) * common, 1e-2)
    return [a, b]


FAMILIES = {
    "Normal": {
        "base_cdf": lambda x, mu, sigma: stats.norm.cdf(x, loc=mu, scale=sigma),
        "init": lambda mean, std: [mean, max(std, 1e-3)],
        "bounds": [(-2.0, 3.0), (1e-3, 5.0)],
        "param_names": ["mu", "sigma"],
        "truncate": True,
    },
    "Exponential": {
        "base_cdf": lambda x, scale: stats.expon.cdf(x, loc=0.0, scale=scale),
        "init": lambda mean, std: [max(mean, 1e-3)],
        "bounds": [(1e-3, 5.0)],
        "param_names": ["scale (1/lambda)"],
        "truncate": True,
    },
    "Logistic": {
        "base_cdf": lambda x, mu, s: stats.logistic.cdf(x, loc=mu, scale=s),
        "init": lambda mean, std: [mean, max(std * np.sqrt(3) / np.pi, 1e-3)],
        "bounds": [(-2.0, 3.0), (1e-3, 5.0)],
        "param_names": ["mu", "s"],
        "truncate": True,
    },
    "Beta": {
        "base_cdf": lambda x, a, b: stats.beta.cdf(x, a, b),
        "init": lambda mean, std: beta_moment_init(mean, std),
        "bounds": [(1e-2, 200.0), (1e-2, 200.0)],
        "param_names": ["alpha", "beta"],
        "truncate": False,
    },
}


def cdf_extra_stats(x: np.ndarray, f: np.ndarray, mean: float, var: float) -> dict:
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

    gaussian_entropy = float(0.5 * np.log(2 * np.pi * np.e * max(var, eps)))
    entropy_gap = gaussian_entropy - entropy

    return {
        "median": median, "mode": mode, "entropy": entropy,
        "auc": auc, "gaussian_entropy": gaussian_entropy, "entropy_gap": entropy_gap,
    }


def fit_family(name, spec, x, f_empirical, mean, std):
    init = spec["init"](mean, std)

    def candidate_cdf(params):
        if spec["truncate"]:
            return truncated_cdf(spec["base_cdf"], x, params)
        return spec["base_cdf"](x, *params)

    def loss(params):
        return float(np.sum((candidate_cdf(params) - f_empirical) ** 2))

    result = minimize(loss, x0=init, method="L-BFGS-B", bounds=spec["bounds"])
    params = result.x
    f_fit = candidate_cdf(params)
    sse = float(np.sum((f_fit - f_empirical) ** 2))
    ks = float(np.max(np.abs(f_fit - f_empirical)))
    return {
        "family": name,
        "params": dict(zip(spec["param_names"], [float(p) for p in params])),
        "sse": sse, "ks": ks, "fitted_curve": f_fit,
    }


def build_report(all_results: dict, output_dir: Path, available_datasets: list[str]) -> None:
    lines = [
        "# Parametric distribution fits to the spectral CDF (small CNN vs. frozen ResNet-18)",
        "",
        "## Full SRF (Spectral Response Function) descriptors by dataset and network",
        "",
        "| Network | Dataset | E[r] | std[r] | Median | Mode | AUC (=1-E[r]) | Entropy | Equivalent Normal entropy | Entropy gap |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (network_name, dataset), data in all_results.items():
        lines.append(
            f"| {network_name} | {DISPLAY[dataset]} | {data['mean']:.4f} | {data['std']:.4f} | "
            f"{data['median']:.4f} | {data['mode']:.4f} | {data['auc']:.4f} | {data['entropy']:.4f} | "
            f"{data['gaussian_entropy']:.4f} | {data['entropy_gap']:.4f} |"
        )
    lines += [
        "",
        "`E[r]` is the recommended reference statistic (a single number per dataset/network that summarizes "
        "\"how much spectrum\" that combination needs). The `entropy gap` (entropy of a Normal with the same "
        "variance minus the actual entropy) is a shape diagnostic: high values indicate that the actual SRF is "
        "more structured/complex (e.g. a plateau or bimodality) than the pair (E[r], std[r]) alone would "
        "capture — this happens notably for ResNet-18/MNIST.",
        "",
        "## Spectral mismatch index (E[r]_ResNet-18 / E[r]_CNN) by dataset",
        "",
        "| Dataset | E[r] CNN | E[r] ResNet-18 | Index |",
        "|---|---|---|---|",
    ]
    cnn_name, resnet_name = list(NETWORKS.keys())
    for dataset in available_datasets:
        e_cnn = all_results[(cnn_name, dataset)]["mean"]
        e_resnet = all_results[(resnet_name, dataset)]["mean"]
        lines.append(f"| {DISPLAY[dataset]} | {e_cnn:.4f} | {e_resnet:.4f} | {e_resnet / e_cnn:.2f} |")

    lines += ["", "## Best family by dataset and network", "", "| Network | Dataset | Best family | Parameters | KS | SSE |", "|---|---|---|---|---|---|"]
    for (network_name, dataset), data in all_results.items():
        best = data["fits"][0]
        params_str = ", ".join(f"{k}={v:.3f}" for k, v in best["params"].items())
        lines.append(f"| {network_name} | {DISPLAY[dataset]} | {best['family']} | {params_str} | {best['ks']:.4f} | {best['sse']:.5f} |")

    lines += ["", "## SSE by family (lower = better fit)", "", "| Network | Dataset | Normal | Exponential | Logistic | Beta |", "|---|---|---|---|---|---|"]
    for (network_name, dataset), data in all_results.items():
        sse_by_family = {f["family"]: f["sse"] for f in data["fits"]}
        lines.append(
            f"| {network_name} | {DISPLAY[dataset]} | {sse_by_family['Normal']:.5f} | "
            f"{sse_by_family['Exponential']:.5f} | {sse_by_family['Logistic']:.5f} | {sse_by_family['Beta']:.5f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    missing: list[tuple[str, str]] = []
    for network_name, base_dir in NETWORKS.items():
        for dataset in DATASETS:
            try:
                x, f_empirical = load_empirical_cdf(base_dir, dataset)
            except FileNotFoundError as exc:
                print(f"WARNING: skipping {network_name} / {DISPLAY[dataset]}: {exc}")
                missing.append((network_name, dataset))
                continue
            mean = float(_trapz(1.0 - f_empirical, x))
            var = float(2.0 * _trapz(x * (1.0 - f_empirical), x) - mean ** 2)
            std = float(np.sqrt(max(var, 0.0)))
            extra = cdf_extra_stats(x, f_empirical, mean, var)

            fits = []
            for family_name, spec in FAMILIES.items():
                fits.append(fit_family(family_name, spec, x, f_empirical, mean, std))
            fits.sort(key=lambda r: r["sse"])
            all_results[(network_name, dataset)] = {
                "mean": mean, "std": std, **extra, "fits": fits, "x": x, "f_empirical": f_empirical,
            }


    available_datasets = [
        dataset for dataset in DATASETS
        if all((network_name, dataset) in all_results for network_name in NETWORKS)
    ]
    if missing:
        print(f"\n{len(missing)} (network, dataset) combination(s) were skipped because their "
              f"*_metrics.json file was not found (the run likely did not finish on Kaggle). "
              f"Comparisons below are limited to: {[DISPLAY[d] for d in available_datasets]}")
    if not available_datasets:
        raise RuntimeError(
            "No dataset is available for all networks -- nothing to compare. "
            "Re-run the missing sweep experiment(s) on Kaggle (with GPU enabled) until they "
            "finish all 4 datasets, then re-run this script."
        )

    print(f"{'Network':<22}{'Dataset':<16}{'E[r]':<8}{'std[r]':<8}{'entropy':<10}{'gap H':<10}")
    print("-" * 64)
    for (network_name, dataset), data in all_results.items():
        print(f"{network_name:<22}{DISPLAY[dataset]:<16}{data['mean']:<8.4f}{data['std']:<8.4f}{data['entropy']:<10.4f}{data['entropy_gap']:<10.4f}")

    print("\nSpectral mismatch index (E[r] ResNet-18 / E[r] CNN):")
    cnn_name, resnet_name = list(NETWORKS.keys())
    for dataset in available_datasets:
        e_cnn = all_results[(cnn_name, dataset)]["mean"]
        e_resnet = all_results[(resnet_name, dataset)]["mean"]
        print(f"  {DISPLAY[dataset]:<16} index={e_resnet/e_cnn:.2f}")

    print(f"\n{'Network':<22}{'Dataset':<16}{'Best family':<14}{'Parameters':<38}{'KS':<8}{'SSE':<10}")
    print("-" * 108)
    for (network_name, dataset), data in all_results.items():
        best = data["fits"][0]
        params_str = ", ".join(f"{k}={v:.3f}" for k, v in best["params"].items())
        print(f"{network_name:<22}{DISPLAY[dataset]:<16}{best['family']:<14}{params_str:<38}{best['ks']:<8.4f}{best['sse']:<10.5f}")

    print("\n\nSSE detail by family (lower = better fit):")
    print(f"{'Network':<22}{'Dataset':<16}{'Normal':<10}{'Exponential':<13}{'Logistic':<12}{'Beta':<10}")
    print("-" * 83)
    for (network_name, dataset), data in all_results.items():
        sse_by_family = {f["family"]: f["sse"] for f in data["fits"]}
        print(f"{network_name:<22}{DISPLAY[dataset]:<16}"
              f"{sse_by_family['Normal']:<10.5f}{sse_by_family['Exponential']:<13.5f}"
              f"{sse_by_family['Logistic']:<12.5f}{sse_by_family['Beta']:<10.5f}")

    for dataset in available_datasets:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, network_name in zip(axes, NETWORKS.keys()):
            data = all_results[(network_name, dataset)]
            ax.plot(data["x"], data["f_empirical"], "k.", ms=4, alpha=0.5, label="Empirical SRF (mean of 3 seeds)")
            colors = {"Normal": "C0", "Exponential": "C1", "Logistic": "C2", "Beta": "C3"}
            for fit in data["fits"]:
                ax.plot(data["x"], fit["fitted_curve"], color=colors[fit["family"]],
                        label=f"{fit['family']} (SSE={fit['sse']:.3f})", linewidth=1.6)
            ax.set_xlabel(f"r — {network_name}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("SRF(r)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"fit_comparison_{dataset}.png", dpi=150)
        plt.close()

    build_report(all_results, OUTPUT_DIR, available_datasets)
    print(f"\nSaved to: {OUTPUT_DIR.resolve()}")
    return all_results


if __name__ == "__main__":
    main()
