"""
Assignment 2, questions 9 and 10: training an MLP with vugrad.

Examples:
    python experiments.py -D synth -l 0.1 --act sigmoid --name synth_sigmoid
    python experiments.py -D synth -l 0.1 --act relu    --name synth_relu
    python experiments.py -D mnist -l 0.0001 --act relu --hidden-mult 2 --name mnist_relu

Each run saves its results to results/<name>.json.
"""
import argparse
import json
import os
import time

import numpy as np
import vugrad as vg
from vugrad.core import Op, TensorNode


# ---------------------------------------------------------------- Question 9
class ReLU(Op):
    """
    Element-wise ReLU: relu(x) = max(0, x).
    """
    @staticmethod
    def forward(context, input):
        context['mask'] = input > 0          # where the input was positive
        return np.maximum(0, input)

    @staticmethod
    def backward(context, goutput):
        # f'(x) is 1 where x > 0 and 0 elsewhere, times goutput (Question 2)
        return goutput * context['mask']


def relu(x):
    return ReLU.do_forward(x)


# ---------------------------------------------------------------- Question 10
class Linear(vg.Linear):
    """
    vugrad's Linear layer, with a choice of weight initialization.
    glorot (default in vugrad), zeros, or normal (standard normal).
    """
    def __init__(self, input_size, output_size, init='glorot'):
        super().__init__(input_size, output_size)
        if init == 'zeros':
            self.w.value[...] = 0.0
        elif init == 'normal':
            self.w.value[...] = np.random.randn(output_size, input_size)
        elif init != 'glorot':
            raise ValueError(f'Unknown init {init}')


class MLP(vg.Module):
    """
    MLP with `layers` hidden layers of size hidden_mult * input_size,
    a choice of nonlinearity, and a log-softmax output.
    """
    def __init__(self, input_size, output_size, hidden_mult=4, layers=1, act='sigmoid', init='glorot'):
        super().__init__()
        hidden_size = hidden_mult * input_size
        sizes = [input_size] + [hidden_size] * layers + [output_size]
        self.layers = [Linear(a, b, init=init) for a, b in zip(sizes[:-1], sizes[1:])]
        self.act = {'sigmoid': vg.sigmoid, 'relu': relu}[act]

    def forward(self, input):
        h = input
        for layer in self.layers[:-1]:
            h = self.act(layer(h))
        return vg.logsoftmax(self.layers[-1](h))

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ---------------------------------------------------------------- training
def accuracy(model, x, y):
    o = model(TensorNode(x))
    acc = (np.argmax(o.value, axis=1) == y).mean()
    o.clear()
    return float(acc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-D', '--dataset', default='synth', choices=['synth', 'mnist'])
    p.add_argument('-b', '--batch-size', type=int, default=128)
    p.add_argument('-e', '--epochs', type=int, default=20)
    p.add_argument('-l', '--learning-rate', type=float, default=0.01)
    p.add_argument('--act', default='sigmoid', choices=['sigmoid', 'relu'])
    p.add_argument('--hidden-mult', type=int, default=4)
    p.add_argument('--layers', type=int, default=1, help='number of hidden layers')
    p.add_argument('--momentum', type=float, default=0.0)
    p.add_argument('--init', default='glorot', choices=['glorot', 'zeros', 'normal'])
    p.add_argument('--scale', action='store_true', help='divide MNIST pixels by 255')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--name', default=None, help='name of the results file')
    args = p.parse_args()

    np.random.seed(args.seed)

    if args.dataset == 'synth':
        (xtrain, ytrain), (xval, yval), num_classes = vg.load_synth()
    else:
        (xtrain, ytrain), (xval, yval), num_classes = vg.load_mnist(final=False, flatten=True)
        if args.scale:
            xtrain, xval = xtrain / 255.0, xval / 255.0

    n, num_features = xtrain.shape
    model = MLP(num_features, num_classes, hidden_mult=args.hidden_mult,
                layers=args.layers, act=args.act, init=args.init)
    velocity = [np.zeros_like(prm.value) for prm in model.parameters()]

    history = {'epoch': [], 'val_acc': [], 'train_loss': []}
    start = time.time()

    for epoch in range(args.epochs):
        total_loss = 0.0
        for fr in range(0, n, args.batch_size):
            to = min(fr + args.batch_size, n)
            batch = TensorNode(xtrain[fr:to])

            loss = vg.logceloss(model(batch), ytrain[fr:to])
            total_loss += float(loss.value)
            loss.backward()

            # gradient descent, with optional momentum
            for prm, v in zip(model.parameters(), velocity):
                v *= args.momentum
                v -= args.learning_rate * prm.grad
                prm.value += v

            loss.zero_grad()
            loss.clear()

        val_acc = accuracy(model, xval, yval)
        history['epoch'].append(epoch + 1)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(total_loss / n)
        print(f'epoch {epoch + 1:3d}   train loss {total_loss / n:.4f}   val acc {val_acc:.4f}')

    seconds = time.time() - start
    print(f'final validation accuracy: {history["val_acc"][-1]:.4f}   ({seconds:.0f} s)')

    name = args.name or f'{args.dataset}_{args.act}'
    os.makedirs('results', exist_ok=True)
    with open(os.path.join('results', f'{name}.json'), 'w') as f:
        json.dump({'args': vars(args), 'seconds': seconds, **history}, f, indent=1)


if __name__ == '__main__':
    main()
