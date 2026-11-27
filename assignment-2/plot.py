"""
Plot validation accuracy and training loss per epoch for several runs.

Example:
    python plot.py q9_sigmoid_vs_relu mnist_sigmoid mnist_relu
saves plots/q9_sigmoid_vs_relu.pdf, using results/mnist_sigmoid.json and results/mnist_relu.json
"""
import json
import os
import sys

import matplotlib.pyplot as plt


def main():
    out, names = sys.argv[1], sys.argv[2:]
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(10, 3.8))

    for name in names:
        with open(os.path.join('results', f'{name}.json')) as f:
            r = json.load(f)
        ax_acc.plot(r['epoch'], r['val_acc'], marker='o', markersize=3, label=name)
        ax_loss.plot(r['epoch'], r['train_loss'], marker='o', markersize=3, label=name)

    ax_acc.set_xlabel('epoch')
    ax_acc.set_ylabel('validation accuracy')
    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('training loss (per instance)')
    ax_loss.set_yscale('log')
    for ax in (ax_acc, ax_loss):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs('plots', exist_ok=True)
    fig.savefig(os.path.join('plots', f'{out}.pdf'))
    print(f'saved plots/{out}.pdf')


if __name__ == '__main__':
    main()
