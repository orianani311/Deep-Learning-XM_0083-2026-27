"""
Appendix A: automatic gradient check for every vugrad Op.

For an Op with inputs X_1, ..., X_r and output Y, we use the loss L = sum(V * Y) for a random V,
so that the gradient of the loss with respect to Y is exactly V. We call the Op's backward with
goutput = V, and compare every entry of the returned gradients with a central finite difference

    dL/dX[idx] ~ (L(X + eps e_idx) - L(X - eps e_idx)) / (2 eps).

Run from the assignment-2 folder:  python gradcheck.py
"""
import numpy as np

from vugrad.core import Add, Sub, Multiply, MatrixMultiply
from vugrad.ops import (Log, Id, Exp, Sum, RowMax, RowSum, Normalize, BatchMM,
                        Sigmoid, Expand, Select, Squeeze, Unsqueeze)
from experiments import ReLU


def forward(op, inputs, kwargs):
    """Run the op's forward on copies of the inputs, with a fresh context."""
    context = {}
    out = op.forward(context, *[x.copy() for x in inputs], **kwargs)
    return context, out


def gradcheck(op, inputs, kwargs=None, eps=1e-6):
    """Return the largest relative error between the analytic and numerical gradients."""
    kwargs = kwargs or {}
    context, out = forward(op, inputs, kwargs)
    V = np.asarray(np.random.randn(*np.shape(out)))

    analytic = op.backward(context, V)
    if not isinstance(analytic, tuple):
        analytic = (analytic,)

    worst = 0.0
    for r, x in enumerate(inputs):
        assert analytic[r].shape == x.shape, f'{op.__name__}: gradient shape {analytic[r].shape} != input shape {x.shape}'
        numeric = np.zeros_like(x, dtype=float)
        for idx in np.ndindex(*x.shape):
            plus = [xi.copy() for xi in inputs]
            minus = [xi.copy() for xi in inputs]
            plus[r][idx] += eps
            minus[r][idx] -= eps
            l_plus = (V * forward(op, plus, kwargs)[1]).sum()
            l_minus = (V * forward(op, minus, kwargs)[1]).sum()
            numeric[idx] = (l_plus - l_minus) / (2 * eps)

        err = np.abs(analytic[r] - numeric).max() / max(np.abs(numeric).max(), 1e-8)
        worst = max(worst, err)
    return worst


def main():
    np.random.seed(0)
    A = lambda *s: np.random.randn(*s)
    P = lambda *s: np.random.rand(*s) + 0.5          # positive values, for Log and Normalize

    cases = [
        ('Add',            Add,            [A(3, 4), A(3, 4)], {}),
        ('Sub',            Sub,            [A(3, 4), A(3, 4)], {}),
        ('Multiply',       Multiply,       [A(3, 4), A(3, 4)], {}),
        ('MatrixMultiply', MatrixMultiply, [A(3, 4), A(4, 2)], {}),
        ('Log',            Log,            [P(3, 4)], {}),
        ('Id',             Id,             [A(3, 4)], {}),
        ('Exp',            Exp,            [A(3, 4)], {}),
        ('Sum',            Sum,            [A(3, 4)], {}),
        ('RowMax',         RowMax,         [A(3, 4)], {}),
        ('RowSum',         RowSum,         [A(3, 4)], {}),
        ('Normalize',      Normalize,      [P(3, 4)], {}),
        ('BatchMM',        BatchMM,        [A(2, 4), A(3, 4)], {}),
        ('Sigmoid',        Sigmoid,        [A(3, 4)], {}),
        ('Expand',         Expand,         [A(1, 4)], {'dim': 0, 'repeats': 3}),
        ('Select',         Select,         [A(3, 4)], {'indices': np.array([0, 3, 1])}),
        ('Squeeze',        Squeeze,        [A(1, 4)], {'dim': 0}),
        ('Unsqueeze',      Unsqueeze,      [A(3, 4)], {'dim': 0}),
        ('ReLU (mine)',    ReLU,           [A(3, 4)], {}),
    ]

    print(f'{"Op":16s} {"max relative error":>20s}')
    for name, op, inputs, kwargs in cases:
        err = gradcheck(op, inputs, kwargs)
        status = 'OK' if err < 1e-6 else 'CHECK'
        print(f'{name:16s} {err:20.2e}   {status}')


if __name__ == '__main__':
    main()
