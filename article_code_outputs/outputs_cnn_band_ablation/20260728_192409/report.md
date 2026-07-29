# CNN band ablation (progressive sweep + complete removal), 4 datasets

## Setup
- Seeds: `[42, 43, 44]`
- Progressive sweep step: `5%`

## MNIST
- Baseline accuracy: `99.32% ± 0.07pp`

| Band removed | Modes removed | Accuracy | Drop |
|---|---|---|---|
| Low | 261 | 34.72% ± 5.94pp | 64.60pp |
| Mid | 268 | 99.27% ± 0.04pp | 0.05pp |
| High | 255 | 99.34% ± 0.06pp | -0.02pp |

## Fashion-MNIST
- Baseline accuracy: `92.55% ± 0.36pp`

| Band removed | Modes removed | Accuracy | Drop |
|---|---|---|---|
| Low | 261 | 34.98% ± 7.18pp | 57.57pp |
| Mid | 268 | 90.44% ± 0.28pp | 2.11pp |
| High | 255 | 91.93% ± 0.34pp | 0.62pp |

## KMNIST
- Baseline accuracy: `96.69% ± 0.19pp`

| Band removed | Modes removed | Accuracy | Drop |
|---|---|---|---|
| Low | 261 | 43.52% ± 11.77pp | 53.17pp |
| Mid | 268 | 96.52% ± 0.26pp | 0.17pp |
| High | 255 | 96.64% ± 0.20pp | 0.04pp |

## CIFAR-10 (grayscale)
- Baseline accuracy: `70.68% ± 1.42pp`

| Band removed | Modes removed | Accuracy | Drop |
|---|---|---|---|
| Low | 341 | 11.42% ± 1.75pp | 59.26pp |
| Mid | 340 | 67.14% ± 1.55pp | 3.54pp |
| High | 343 | 70.14% ± 1.44pp | 0.54pp |
