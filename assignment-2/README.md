# Assignment 2: Automatic differentiation

Backward functions for tensor operations, the `vugrad` autodiff library
(https://github.com/dlvu/vugrad), and an introduction to PyTorch.

## Contents

| File | Description |
|---|---|
| `Assignment_2_Deliallisi.pdf` | The report, with all answers (questions 1 to 12) and an appendix |
| `main.tex` | LaTeX source of the report |
| `assignment_2.ipynb` | Numerical gradient checks (questions 1 to 4) and vugrad exploration (questions 5 to 7) |
| `experiments.py` | ReLU op and MLP experiments in vugrad (questions 9 and 10) |
| `pytorch_cifar.py` | CIFAR-10 classifier in PyTorch (questions 11 and 12) |
| `gradcheck.py` | Automatic gradient check of all vugrad Ops (appendix A.1) |
| `seeds_summary.py` | Mean and standard deviation over seeds (appendix A.2) |
| `plot.py` | Plots validation accuracy and training loss from `results/` |
| `results/` | Results of every run (JSON) |
| `plots/` | The figures used in the report |

## Overview

- **Questions 1 to 4:** derivations of backward functions, checked numerically with finite differences.
- **Questions 5 to 8:** how vugrad builds and traverses the computation graph; proof of the `Normalize` backward.
- **Question 9:** ReLU op; ReLU reaches 96.12% validation accuracy on MNIST vs 90.42% for sigmoid.
- **Question 10:** width, depth, momentum and initialization.
- **Questions 11 and 12:** hyperparameter search and other changes for the blitz network on CIFAR-10;
  best setting (batch norm + data augmentation) reaches **67.54% test accuracy**.
- **Appendix:** gradient check of every vugrad Op, and seed variation for question 10.

## Running the code

Install vugrad (`pip install -e .` in its repository), numpy, matplotlib, torch and torchvision.
Run the scripts from inside this folder; the exact commands are in the report.
MNIST and CIFAR-10 are downloaded on the first run and are not included in this repository.
