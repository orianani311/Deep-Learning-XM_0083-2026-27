"""
Appendix B: mean and standard deviation of the final validation accuracy over seeds.

Uses results/<name>.json for seed 0 (from question 10) and results/<name>_s<seed>.json for the other seeds.
Run from the assignment-2 folder:  python seeds_summary.py
"""
import json
import os

import numpy as np

RUNS = ['relu', 'wider', 'deeper', 'momentum']
SEEDS = [0, 1, 2]

print(f'{"run":10s} {"seed 0":>8s} {"seed 1":>8s} {"seed 2":>8s} {"mean":>8s} {"std":>7s}')
for run in RUNS:
    accs = []
    for seed in SEEDS:
        name = run if seed == 0 else f'{run}_s{seed}'
        with open(os.path.join('results', f'{name}.json')) as f:
            accs.append(100 * json.load(f)['val_acc'][-1])
    accs = np.array(accs)
    cells = ' '.join(f'{a:8.2f}' for a in accs)
    print(f'{run:10s} {cells} {accs.mean():8.2f} {accs.std(ddof=1):7.2f}')
