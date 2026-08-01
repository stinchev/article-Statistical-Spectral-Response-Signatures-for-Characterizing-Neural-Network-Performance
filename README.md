# Statistical Spectral Response Signatures for Characterizing Neural Networks Performance

Code and results accompanying the paper:

> Stivan Svetoslavov Tinchev, Sonia Rubio Herranz, Antonio López Montes, Fernando Carlos López Hernández.
> **Statistical Spectral Response Signatures for Characterizing Neural Networks Performance.**

Affiliations: Universidad Complutense de Madrid — Faculty of Mathematical Sciences;
Department of Statistics and Operations Research; Department of Applied Mathematics
and Mathematical Analysis; Department of Artificial Intelligence.

## Summary

This work introduces the **Spectral Response Function (SRF)**, a cumulative curve that
describes how the predictive performance of a trained neural network evolves as
progressively larger portions of the Fourier spectrum are made available at its input.
Interpreting the SRF as a cumulative distribution function, we define the **Statistical
Spectral Response Signature (SRS)**: a five-dimensional descriptor — expected spectral
radius E[R], spectral dispersion σ[R], median, mode, and entropy H — that summarizes
the spectral behavior of a trained model.

The framework is evaluated on MNIST, Fashion-MNIST, KMNIST and grayscale CIFAR-10,
comparing a small CNN trained from scratch against a pretrained, frozen ResNet-18.

## Repository structure

```
article_codes/           Python scripts that produce the paper's experiments
article_code_outputs/    Saved outputs (JSON statistics, figures, reports) from each script
```

### Scripts used directly in the paper

| Script | Produces |
|---|---|
| `vision_spectral_cdf_experiment.py` | Figure 1, CNN rows of Table 4 |
| `vision_resnet18_spectral_cdf_experiment.py` | Figure 2, ResNet-18 rows of Table 4 |
| `discrete_srf_statistics_experiment.py` | Table 4, Figure 3 |
| `srf_stability_seeds_experiment.py` | Figure 4, Table 5 |
| `srf_activation_experiment.py` | Figure 5, Table 6 |

### Additional experiments (not referenced in the paper)

| Script | Description |
|---|---|
| `cnn_band_ablation_4datasets_experiment.py` | Band-pruning ablation, small CNN |
| `resnet18_band_ablation_4datasets_experiment.py` | Band-pruning ablation, ResNet-18 |
| `fit_spectral_distributions_experiment.py` | Parametric-family fits (Logistic/Beta/Normal/Exponential) to the empirical SRF |

These three scripts were part of a broader exploration of the SRF methodology and are
kept in the repository for completeness, but their outputs do not appear in this paper.

### Outputs

Each script under `article_codes/` writes its results (JSON with the raw statistics,
`report.md`, and the corresponding figures) to a matching folder under
`article_code_outputs/`, e.g. `vision_spectral_cdf_experiment.py` → `outputs_srf_cnn/`.

## Requirements

See `requirements.txt`. The scripts use two independent stacks depending on the
architecture being trained:

- **TensorFlow/Keras** for the small CNN.
- **PyTorch/torchvision** for the pretrained ResNet-18.

## Running the experiments

Each script can be run locally or on Kaggle. It auto-detects the environment: on
Kaggle it looks for input/output under `/kaggle/working` and `/kaggle/input`; locally
it falls back to a configurable output path at the top of the script. Datasets
(MNIST, Fashion-MNIST, KMNIST, CIFAR-10) are downloaded automatically on first run.

All experiments in Section 3.4 of the paper use three independent random seeds
(42, 43, 44) unless stated otherwise; the stability experiment (Table 5, Figure 4)
uses ten seeds (42–51). For the pretrained ResNet-18, the SRF sweep is evaluated on a
fixed random subset of 2,000 test images (baseline accuracy is still computed on the
full test set) to keep the 150-point spectral sweep computationally tractable.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

See `LICENSE`.
