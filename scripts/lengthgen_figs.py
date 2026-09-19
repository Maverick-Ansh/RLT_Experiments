"""Fig. 4 (length generalization), Fig. 5 (the three S5 scoring rules) and Table 2."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, '/content/RLT_Experiments')
import matplotlib.pyplot as plt
from rlt.viz import (ORDER, COLOR, MARKER, DASH, LABEL, TASK_LABEL, style_axes, legend,
                     reference_line, SURFACE, INK, INK_2, INK_MUTED, TITLE_Y, RECT_TOP)
from rlt import tasks as T

RES = '/content/RLT_Experiments/results/lengthgen'
FIG = '/content/RLT_Experiments/figures'
METRIC = {'addition': 'token', 'parity': 'final', 'mod5_flat': 'final',
          'mod5_brackets': 'final', 's5_swaps': 'final', 's5_standard': 'final'}
try:
    BRACK = {r['task']: r for r in json.load(open('/content/RLT_Experiments/results/eval_brackets.json'))}
except Exception:
    BRACK = {}


def load():
    out = {}
    for f in glob.glob(f'{RES}/*.json'):
        r = json.load(open(f))
        out[(r['task'], r['arch'], r['seed'])] = r
    return out


def series(runs, task, arch, key='final', seeds=(42, 43, 44)):
    """-> lengths, mean, sd  across seeds; None if any seed is missing (App. J.3: no imputation)."""
    per = []
    for s in seeds:
        r = runs.get((task, arch, s))
        if r is None: return None
        per.append({row['length']: row[key] for row in r['rows']})
    L = sorted(set.intersection(*[set(p) for p in per]))
    if not L: return None
    a = np.array([[p[l] for l in L] for p in per], float)
    return np.array(L), a.mean(0), a.std(0, ddof=1)


def fig4(runs, out='fig4_length_generalization.png'):
    fig, axes = plt.subplots(3, 2, figsize=(11, 11.5))
    fig.patch.set_facecolor(SURFACE)
    handles, labels = [], []
    for ax, task in zip(axes.ravel(), T.TASKS):
        top = 0
        for arch in ORDER:
            s = series(runs, task, arch, METRIC[task])
            if s is None: continue
            L, mu, sd = s
            xs = np.arange(len(L))
            top = max(top, (mu + sd).max())
            ln = ax.errorbar(xs, mu, yerr=sd, color=COLOR[arch], linestyle=DASH[arch],
                             linewidth=2, marker=MARKER[arch], markersize=5, capsize=3,
                             elinewidth=1.2, label=LABEL[arch], zorder=3)
            if arch not in labels: handles.append(ln); labels.append(arch)
            ax.set_xticks(xs); ax.set_xticklabels([str(v) for v in L], fontsize=8)
        # shade the trained range
        L = T.TEST_LENGTHS[task]
        n_in = sum(1 for v in L if v <= T.TRAINED_UPTO[task])
        if n_in:
            ax.axvspan(-0.4, n_in - 0.6, color='#efeeea', zorder=0)
            ax.annotate('trained', xy=(max(n_in-1, 0)*0.5, 2), fontsize=7, color=INK_MUTED)
        if T.CHANCE[task]: reference_line(ax, T.CHANCE[task], 'uniform')
        if task in BRACK and BRACK[task]['majority'] > (T.CHANCE[task] or 0) + 0.3:
            reference_line(ax, BRACK[task]['majority'], 'majority-class')
        style_axes(ax, ylabel=('Teacher-forced token accuracy (%)' if task == 'addition'
                               else 'Final-answer accuracy (%)'),
                   xlabel=('Digits in each operand' if task == 'addition'
                           else 'Operations' if task.startswith('s5') else 'Expression tokens'),
                   title=f"{TASK_LABEL[task]}\ntrained: up to {T.TRAINED_UPTO[task]}")
        ax.set_ylim(0, max(5, min(108, top * 1.12)))
    legend(fig, handles, [LABEL[a] for a in labels], ncol=7)
    fig.suptitle('Length generalization from the best in-distribution checkpoint, three seeds',
                 fontsize=12, color=INK, y=TITLE_Y)
    fig.text(0.5, 0.012, 'mean $\\pm$ sample SD (n=3) · identical test examples for every model '
             'and seed · shaded = trained lengths', ha='center', fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.025, 1, RECT_TOP])
    fig.savefig(f'{FIG}/{out}', dpi=150, facecolor=SURFACE); plt.close(fig)
    print('wrote', out)


def fig5(runs, out='fig5_s5_scoring_rules.png'):
    """Rows = prefix-token / final-state / whole-sequence; columns = standard / swaps."""
    rules = [('token', 'Prefix-token accuracy (%)'), ('final', 'Final-state accuracy (%)'),
             ('sequence', 'Whole-sequence accuracy (%)')]
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 11))
    fig.patch.set_facecolor(SURFACE)
    handles, labels = [], []
    for i, (key, ylab) in enumerate(rules):
        for j, task in enumerate(['s5_standard', 's5_swaps']):
            ax = axes[i, j]; top = 0
            for arch in ORDER:
                s = series(runs, task, arch, key)
                if s is None: continue
                L, mu, sd = s
                xs = np.arange(len(L)); top = max(top, (mu + sd).max())
                ln = ax.errorbar(xs, mu, yerr=sd, color=COLOR[arch], linestyle=DASH[arch],
                                 linewidth=2, marker=MARKER[arch], markersize=5, capsize=3,
                                 elinewidth=1.2, label=LABEL[arch], zorder=3)
                if arch not in labels and i == 0 and j == 0:
                    handles.append(ln); labels.append(arch)
                ax.set_xticks(xs); ax.set_xticklabels([str(v) for v in L], fontsize=7.5)
            reference_line(ax, T.CHANCE[task])
            style_axes(ax, ylabel=ylab if j == 0 else None, xlabel='Operations',
                       title=f'{TASK_LABEL[task]}  ·  train: 32 operations')
            ax.set_ylim(0, max(4, min(108, top * 1.15)))
    legend(fig, handles, [LABEL[a] for a in labels], ncol=7)
    fig.suptitle('$S_5$ length generalization under three scoring rules',
                 fontsize=12, color=INK, y=TITLE_Y)
    fig.text(0.5, 0.012, 'whole-sequence requires EVERY prefix prediction to be correct · '
             'panel-specific vertical scales', ha='center', fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.025, 1, RECT_TOP])
    fig.savefig(f'{FIG}/{out}', dpi=150, facecolor=SURFACE); plt.close(fig)
    print('wrote', out)


def table2(runs):
    lines, rowsj = [], []
    hdr = f"{'Task (test length)':<26}" + ''.join(f'{LABEL[a]:>24}' for a in ORDER)
    lines += [hdr, '-' * len(hdr)]
    for task in T.TASKS:
        Lmax = T.TEST_LENGTHS[task][-1]
        cells = []
        for arch in ORDER:
            s = series(runs, task, arch, METRIC[task])
            if s is None: cells.append(f"{'--':>24}"); continue
            L, mu, sd = s
            k = int(np.argmax(L))
            cells.append(f'{mu[k]:>16.2f} ± {sd[k]:5.2f}')
            rowsj.append(dict(task=task, arch=arch, length=int(L[k]), mean=float(mu[k]),
                              sd=float(sd[k]), metric=METRIC[task]))
        lines.append(f'{task + f" ({Lmax})":<26}' + ''.join(cells))
    txt = '\n'.join(lines)
    open(f'{FIG}/table2.txt', 'w').write(txt)
    json.dump(rowsj, open('/content/RLT_Experiments/results/table2.json', 'w'), indent=1)
    return txt


if __name__ == '__main__':
    runs = load()
    print(f'{len(runs)} length-generalization records\n')
    print(table2(runs))
    fig4(runs); fig5(runs)
