"""Bracket every metric BEFORE the sweep: uniform floor, majority-class shortcut, untrained net.

If a reported accuracy does not clear the shortcut row, the model has not learned the task -- it
has learned the label prior.  Run this first; quote it in every results table.
"""
import sys, json, collections
import numpy as np, torch
sys.path.insert(0, '/content/RLT_Experiments')
from rlt import tasks as T, data as D
from rlt.build import build
from rlt.train import metrics

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
rows = []
print(f"{'task':<15}{'uniform':>9}{'majority':>10}{'untrained':>11}{'classes':>9}{'metric':>15}")
print('-' * 70)
for task in T.TASKS:
    L = T.VAL_LENGTHS[task][0]
    seed = T.ADD_SEED + 1 if task == 'addition' else T.VAL_SEED_FORMAL
    x, y, m = T.make(task, 1024, seed, length=L)

    # ---- majority-class shortcut, measured on the ACTUAL labels of the held-out set ----
    if task.startswith('s5'):
        labs = y[m]                                     # every supervised prefix state
        metric_name = 'prefix-token'
    elif task == 'addition':
        labs = y[m]                                     # every answer token incl. EOS
        metric_name = 'answer-token'
    else:
        labs = np.array([y[r][m[r]][-1] for r in range(len(y))])
        metric_name = 'final-label'
    cnt = collections.Counter(labs.tolist())
    majority = 100.0 * max(cnt.values()) / len(labs)
    uniform = 100.0 / len(cnt) if task == 'addition' else T.CHANCE[task]

    # ---- untrained network of the same architecture ----
    torch.manual_seed(0)
    mdl = build('rlt1_4+4', T.VOCAB, d_model=128).to(dev).eval()
    xb = torch.from_numpy(x[:256]).to(dev); yb = torch.from_numpy(y[:256]).to(dev)
    mb = torch.from_numpy(m[:256]).to(dev)
    with torch.no_grad():
        r = metrics(mdl(xb), yb, mb)
    untrained = r['token'] if metric_name != 'final-label' else r['final']

    rows.append(dict(task=task, uniform=uniform, majority=majority, untrained=untrained,
                     metric=metric_name, n_classes=len(cnt)))
    print(f'{task:<15}{uniform:>8.2f}%{majority:>9.2f}%{untrained:>10.2f}%'
          f'{len(cnt):>9}{metric_name:>15}')
    del mdl; torch.cuda.empty_cache()

json.dump(rows, open('/content/RLT_Experiments/results/eval_brackets.json', 'w'), indent=1)
print("""
READING THE TABLE
  uniform   = the paper's dotted line: predict every class with equal probability.
  majority  = ignore the input, always emit the most common label.  This is the number an
              accuracy must BEAT to be evidence of anything.
  untrained = a randomly initialized RLT-1 4+4 at the same width, no training at all.
Where majority > uniform the paper's dotted line is the WRONG floor, and this repo quotes the
majority row instead.""")
