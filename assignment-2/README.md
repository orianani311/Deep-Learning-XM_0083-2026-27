# Assignment 2: Automatic differentiation

Backward functions worked out by hand, backpropagation with the `vugrad`
library, and a classifier in PyTorch.

## Contents

| File | Description |
|---|---|
| `assignment_2.ipynb` | The code for all questions, with outputs |
| `plots/` | The figures used in the report |

## Overview

- **Part 1 (questions 1 to 4):** backward functions for element-wise division,
  element-wise functions, an MLP layer without bias, and broadcasting.
- **Part 2 (questions 5 to 10):** the computation graph in `vugrad`, a ReLU op, and
  a change to the architecture.
- **Part 3 (questions 11 and 12):** a classifier in PyTorch and hyperparameter
  experiments.

## Running the code

Part 2 needs [vugrad](https://github.com/dlvu/vugrad): clone it and run
`pip install -e .` in its folder. Then run the notebook cells from top to bottom.
