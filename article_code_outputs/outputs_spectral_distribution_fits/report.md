# Parametric distribution fits to the spectral CDF (small CNN vs. frozen ResNet-18)

## Full SRF (Spectral Response Function) descriptors by dataset and network

| Network | Dataset | E[r] | std[r] | Median | Mode | AUC (=1-E[r]) | Entropy | Equivalent Normal entropy | Entropy gap |
|---|---|---|---|---|---|---|---|---|---|
| Small CNN | MNIST | 0.1325 | 0.0483 | 0.1307 | 0.1070 | 0.8675 | -1.9368 | -1.6118 | 0.3250 |
| Small CNN | Fashion-MNIST | 0.1618 | 0.1144 | 0.1334 | 0.1070 | 0.8382 | -1.2843 | -0.7489 | 0.5353 |
| Small CNN | KMNIST | 0.1453 | 0.0602 | 0.1473 | 0.1556 | 0.8547 | -1.7190 | -1.3904 | 0.3286 |
| Small CNN | CIFAR-10 (grayscale) | 0.2635 | 0.1267 | 0.2660 | 0.2670 | 0.7365 | -0.8818 | -0.6472 | 0.2346 |
| Frozen ResNet-18 | MNIST | 0.5141 | 0.1890 | 0.5579 | 0.6576 | 0.4859 | -1.3659 | -0.2469 | 1.1190 |
| Frozen ResNet-18 | Fashion-MNIST | 0.3643 | 0.2380 | 0.3951 | 0.5567 | 0.6357 | -1.0358 | -0.0167 | 1.0191 |
| Frozen ResNet-18 | KMNIST | 0.4535 | 0.1890 | 0.4322 | 0.3553 | 0.5465 | -0.7831 | -0.2473 | 0.5358 |
| Frozen ResNet-18 | CIFAR-10 (grayscale) | 0.4379 | 0.1476 | 0.4386 | 0.4364 | 0.5621 | -0.9135 | -0.4946 | 0.4189 |

`E[r]` is the recommended reference statistic (a single number per dataset/network that summarizes "how much spectrum" that combination needs). The `entropy gap` (entropy of a Normal with the same variance minus the actual entropy) is a shape diagnostic: high values indicate that the actual SRF is more structured/complex (e.g. a plateau or bimodality) than the pair (E[r], std[r]) alone would capture — this happens notably for ResNet-18/MNIST.

## Spectral mismatch index (E[r]_ResNet-18 / E[r]_CNN) by dataset

| Dataset | E[r] CNN | E[r] ResNet-18 | Index |
|---|---|---|---|
| MNIST | 0.1325 | 0.5141 | 3.88 |
| Fashion-MNIST | 0.1618 | 0.3643 | 2.25 |
| KMNIST | 0.1453 | 0.4535 | 3.12 |
| CIFAR-10 (grayscale) | 0.2635 | 0.4379 | 1.66 |

## Best family by dataset and network

| Network | Dataset | Best family | Parameters | KS | SSE |
|---|---|---|---|---|---|
| Small CNN | MNIST | Logistic | mu=0.132, s=0.026 | 0.0775 | 0.01775 |
| Small CNN | Fashion-MNIST | Logistic | mu=0.086, s=0.087 | 0.1094 | 0.02198 |
| Small CNN | KMNIST | Logistic | mu=0.141, s=0.033 | 0.0667 | 0.01605 |
| Small CNN | CIFAR-10 (grayscale) | Logistic | mu=0.254, s=0.074 | 0.0563 | 0.01545 |
| Frozen ResNet-18 | MNIST | Logistic | mu=0.548, s=0.086 | 0.1322 | 0.45114 |
| Frozen ResNet-18 | Fashion-MNIST | Beta | alpha=1.184, beta=1.932 | 0.1023 | 0.22917 |
| Frozen ResNet-18 | KMNIST | Beta | alpha=2.370, beta=2.899 | 0.0528 | 0.04011 |
| Frozen ResNet-18 | CIFAR-10 (grayscale) | Logistic | mu=0.439, s=0.080 | 0.0474 | 0.03600 |

## SSE by family (lower = better fit)

| Network | Dataset | Normal | Exponential | Logistic | Beta |
|---|---|---|---|---|---|
| Small CNN | MNIST | 0.01915 | 0.57441 | 0.01775 | 0.01875 |
| Small CNN | Fashion-MNIST | 0.03335 | 0.12847 | 0.02198 | 0.02294 |
| Small CNN | KMNIST | 0.01856 | 0.56152 | 0.01605 | 0.01652 |
| Small CNN | CIFAR-10 (grayscale) | 0.02662 | 1.33768 | 0.01545 | 0.03465 |
| Frozen ResNet-18 | MNIST | 0.49209 | 4.02842 | 0.45114 | 0.48250 |
| Frozen ResNet-18 | Fashion-MNIST | 0.26341 | 0.47076 | 0.29252 | 0.22917 |
| Frozen ResNet-18 | KMNIST | 0.06511 | 1.28832 | 0.07583 | 0.04011 |
| Frozen ResNet-18 | CIFAR-10 (grayscale) | 0.05837 | 3.85506 | 0.03600 | 0.06715 |