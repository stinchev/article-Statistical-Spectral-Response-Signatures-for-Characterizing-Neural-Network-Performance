# Frozen ResNet-18 band ablation (progressive sweep + complete removal), 4 datasets

## Setup
- Seeds: `[42, 43, 44]`
- Progressive sweep step: `5%`
- Test subset for the progressive sweep: `2000`

## MNIST
- Baseline accuracy (full test set): `97.76% ± 0.05pp`

| Band removed | Modes removed | Accuracy | Drop |
|---|---|---|---|
| Low | 261 | 16.88% ± 0.37pp | 80.88pp |
| Mid | 268 | 29.47% ± 0.54pp | 68.29pp |
| High | 255 | 82.64% ± 1.20pp | 15.12pp |

## Fashion-MNIST
- Baseline accuracy (full test set): `89.18% ± 0.12pp`

| Band removed | Modes removed | Accuracy | Drop |
|---|---|---|---|
| Low | 261 | 29.22% ± 0.78pp | 59.96pp |
| Mid | 268 | 53.45% ± 0.39pp | 35.72pp |
| High | 255 | 77.09% ± 0.82pp | 12.08pp |

## KMNIST
- Baseline accuracy (full test set): `83.87% ± 0.55pp`

| Band removed | Modes removed | Accuracy | Drop |
|---|---|---|---|
| Low | 261 | 13.10% ± 0.14pp | 70.76pp |
| Mid | 268 | 42.42% ± 0.33pp | 41.45pp |
| High | 255 | 70.72% ± 0.85pp | 13.14pp |

## CIFAR-10 (grayscale)
- Baseline accuracy (full test set): `79.25% ± 0.18pp`

| Band removed | Modes removed | Accuracy | Drop |
|---|---|---|---|
| Low | 341 | 14.69% ± 0.32pp | 64.57pp |
| Mid | 340 | 52.13% ± 0.17pp | 27.13pp |
| High | 343 | 73.10% ± 0.08pp | 6.16pp |
