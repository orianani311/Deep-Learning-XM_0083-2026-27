# Assignment 1: Implementing an MLP using NumPy

A small multilayer perceptron (sigmoid hidden layer, softmax output, cross-entropy
loss) implemented from scratch, first in plain Python and then vectorized with
NumPy, and trained on a synthetic dataset and on MNIST.

## Contents

| File | Description |
|---|---|
| `Assignment_1_Deliallisi.pdf` | The report, with all answers (questions 1 to 8) and an appendix |
| `assignment_1.ipynb` | The code for all questions, with outputs |
| `data.py` | Data loaders for the synthetic data and MNIST (provided by the course) |
| `plots/` | The figures used in the report |

## Overview

- **Questions 1 and 2:** derivatives of the softmax and cross-entropy loss.
- **Question 3:** forward and backward pass of a small network in plain Python.
- **Question 4:** training on the synthetic data with stochastic gradient descent.
- **Questions 5 and 6:** vectorized forward and backward pass in NumPy, trained on MNIST.
- **Question 7:** batched implementation.
- **Question 8:** training vs. validation loss, random initializations, learning rate
  search and a final run on the test set (**97.46% test accuracy**).
- **Appendix:** comparison of sigmoid, tanh and ReLU activations.

## Running the code

Place `data.py` next to the notebook and run the cells from top to bottom.
`load_mnist()` downloads MNIST on the first call and caches it as `mnist.pkl`
(not included in this repository).
