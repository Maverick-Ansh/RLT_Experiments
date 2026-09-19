"""Rebuild the paper's training figures and tables from results/main/.

Seed aggregation follows App. J.3 exactly:
    mean_{m,t} = (1/3) sum_s a_{m,s,t}
    SD_{m,t}   = sqrt( (1/2) sum_s (a_{m,s,t} - mean)^2 )      <- sample SD, denominator n-1
"Every aggregate uses all three seeds at the same step, without interpolation or missing-seed
imputation" -- so a model-step with fewer than 3 seeds present is dropped, never averaged over 2.
"""
import glob, json, os, sys, math
import numpy as np
sys.path.insert(0, '/content/RLT_Experiments')
import matplotlib.pyplot as plt
from rlt.viz import (ORDER, COLOR, MARKER, DASH, LABEL, TASK_LABEL, style_axes, new_fig,
                     legend, reference_line, SURFACE, INK, INK_2, INK_MUTED, TITLE_Y, RECT_TOP)
from rlt import tasks as T

RES = '/content/RLT_Experiments/results/main'
FIG = '/content/RLT_Experiments/figures'
os.makedirs(FIG, exist_ok=True)

# Which metric each task reports, per App. J.3.
METRIC = {'addition': 'token',          # teacher-forced answer-token accuracy
          'parity': 'final', 'mod5_flat': 'final', 'mod5_brackets': 'final',
          's5_swaps': 'final', 's5_standard': 'final'}      # final-state accuracy

try:
    BRACK = {r['task']: r for r in json.load(open('/content/RLT_Experiments/results/eval_brackets.json'))}
except Exception:
    BRACK = {}


def load():
    runs = {}
    for f in glob.glob(f'{RES}/*.json'):
        r = json.load(open(f))
        runs[(r['task'], r['arch'], r['seed'])] = r
    return runs


def curve(runs, task, arch, key='final', seeds=(42, 43, 44)):
    """-> steps, mean, sd, per_seed   (only steps where ALL seeds are present)"""
    per = []
    for s in seeds:
        r = runs.get((task, arch, s))
        if r is None:
            return None
        per.append({h['step']: h[key] for h in r['history']})
    steps = sorted(set.intersection(*[set(p) for p in per]))
    if not steps:
        return None
    a = np.array([[p[t] for t in steps] for p in per], dtype=float)   # (3, S)
    return np.array(steps), a.mean(0), a.std(0, ddof=1), a


def fig2(runs, out='fig2_training_curves.png'):
    fig, axes = plt.subplots(3, 2, figsize=(11, 11.5))
    fig.patch.set_facecolor(SURFACE)
    handles, labels = [], []
    for ax, task in zip(axes.ravel(), T.TASKS):
        key = METRIC[task]
        top = 0
        for arch in ORDER:
            c = curve(runs, task, arch, key)
            if c is None: continue
            st, mu, sd, _ = c
            top = max(top, (mu + sd).max())
            ln, = ax.plot(st, mu, color=COLOR[arch], linestyle=DASH[arch], linewidth=2,
                          marker=MARKER[arch], markersize=4, markevery=max(1, len(st)//8),
                          label=LABEL[arch], zorder=3)
            ax.fill_between(st, np.clip(mu-sd, 0, 100), np.clip(mu+sd, 0, 100),
                            color=COLOR[arch], alpha=0.13, linewidth=0, zorder=2)
            if arch not in labels:
                handles.append(ln); labels.append(arch)
        style_axes(ax, ylabel=('Teacher-forced token accuracy (%)' if task == 'addition'
                               else 'Final-answer accuracy (%)'),
                   xlabel='Optimizer step (global batch 512)', title=TASK_LABEL[task])
        if T.CHANCE[task]:
            reference_line(ax, T.CHANCE[task], 'uniform')
        if task in BRACK and BRACK[task]['majority'] > (T.CHANCE[task] or 0) + 0.3:
            reference_line(ax, BRACK[task]['majority'], 'majority-class')
        ax.set_ylim(0, max(5, min(105, top * 1.12)))
    legend(fig, [h for h in handles], [LABEL[a] for a in labels], ncol=7)
    fig.suptitle('Validation accuracy during training, three seeds for every task',
                 fontsize=12, color=INK, y=TITLE_Y)
    fig.text(0.5, 0.012, 'mean $\\pm$ sample SD over seeds 42, 43, 44 (n=3) · unsmoothed · '
             'dotted lines are the uniform and majority-class floors',
             ha='center', fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.025, 1, RECT_TOP])
    fig.savefig(f'{FIG}/{out}', dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print('wrote', out)


def table1(runs, step=None):
    """Table 1: accuracy at the final step, mean +- sample SD."""
    lines, rowsj = [], []
    hdr = f"{'Task':<22}" + ''.join(f'{LABEL[a]:>24}' for a in ORDER)
    lines.append(hdr); lines.append('-' * len(hdr))
    for task in T.TASKS:
        cells = []
        for arch in ORDER:
            c = curve(runs, task, arch, METRIC[task])
            if c is None:
                cells.append(f"{'--':>24}"); continue
            st, mu, sd, _ = c
            i = -1 if step is None else int(np.argmin(np.abs(st - step)))
            cells.append(f'{mu[i]:>16.2f} ± {sd[i]:5.2f}')
            rowsj.append(dict(task=task, arch=arch, step=int(st[i]),
                              mean=float(mu[i]), sd=float(sd[i]), metric=METRIC[task]))
        lines.append(f'{task:<22}' + ''.join(cells))
    txt = '\n'.join(lines)
    open(f'{FIG}/table1.txt', 'w').write(txt)
    json.dump(rowsj, open('/content/RLT_Experiments/results/table1.json', 'w'), indent=1)
    return txt


def fig3(runs, task='parity', steps=(250, 500), out='fig3_parity_checkpoints.png'):
    """Fig. 3: mean +- SD with individual seeds shown as crosses."""
    fig, axes = plt.subplots(1, len(steps), figsize=(11, 4.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, S in zip(np.atleast_1d(axes), steps):
        for i, arch in enumerate(ORDER):
            c = curve(runs, task, arch, METRIC[task])
            if c is None: continue
            st, mu, sd, a = c
            j = int(np.argmin(np.abs(st - S)))
            ax.errorbar(i, mu[j], yerr=sd[j], fmt=MARKER[arch], color=COLOR[arch],
                        markersize=7, capsize=4, elinewidth=1.6, zorder=3)
            ax.scatter([i + 0.22] * a.shape[0], a[:, j], marker='x', s=26,
                       color=COLOR[arch], alpha=0.85, zorder=3, linewidths=1.2)
            ax.annotate(f'{mu[j]:.1f}±{sd[j]:.1f}', (i, 108), ha='center', fontsize=7,
                        color=INK_2)
        style_axes(ax, ylabel='Validation accuracy (%)' if S == steps[0] else None,
                   title=f'Step {S} · {S*512:,} training examples')
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([LABEL[a].replace(' ', '\n', 1) for a in ORDER], fontsize=7.5)
        if T.CHANCE[task]: reference_line(ax, T.CHANCE[task], 'uniform')
        if task in BRACK: reference_line(ax, BRACK[task]['majority'], 'majority')
        ax.set_ylim(0, 120)
    fig.suptitle(f'{TASK_LABEL[task]} at two training checkpoints', fontsize=12, color=INK)
    fig.text(0.5, 0.015, 'circles/squares: mean ± sample SD across seeds 42, 43, 44 · '
             'crosses: individual seeds', ha='center', fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(f'{FIG}/{out}', dpi=150, facecolor=SURFACE)
    plt.close(fig); print('wrote', out)


def fig13(runs, task='parity', out='fig13_per_seed.png'):
    """Fig. 13: one panel per architecture, one line per seed -- averages hide unstable runs."""
    fig, axes = plt.subplots(4, 2, figsize=(10, 11), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    seed_c = ['#2a78d6', '#eb6834', '#1baf7a']
    for ax, arch in zip(axes.ravel(), ORDER):
        for k, s in enumerate([42, 43, 44]):
            r = runs.get((task, arch, s))
            if r is None: continue
            ax.plot([h['step'] for h in r['history']], [h[METRIC[task]] for h in r['history']],
                    color=seed_c[k], linewidth=1.7, label=f'Seed {s}', zorder=3)
        style_axes(ax, ylabel='Validation accuracy (%)', xlabel='Optimizer step',
                   title=LABEL[arch])
        if T.CHANCE[task]: reference_line(ax, T.CHANCE[task])
        ax.set_ylim(0, 105)
    axes.ravel()[-1].axis('off')
    h, l = axes.ravel()[0].get_legend_handles_labels()
    legend(fig, h, l, ncol=3)
    fig.suptitle(f'{TASK_LABEL[task]}: every initialization seed, unaveraged',
                 fontsize=12, color=INK, y=TITLE_Y)
    fig.tight_layout(rect=[0, 0, 1, RECT_TOP])
    fig.savefig(f'{FIG}/{out}', dpi=150, facecolor=SURFACE)
    plt.close(fig); print('wrote', out)


def fig15(runs, out='fig15_training_loss.png'):
    fig, axes = plt.subplots(3, 2, figsize=(11, 11))
    fig.patch.set_facecolor(SURFACE)
    handles, labels = [], []
    for ax, task in zip(axes.ravel(), T.TASKS):
        for arch in ORDER:
            c = curve(runs, task, arch, 'train_loss')
            if c is None: continue
            st, mu, sd, _ = c
            ln, = ax.plot(st, np.maximum(mu, 1e-8), color=COLOR[arch], linestyle=DASH[arch],
                          linewidth=2, label=LABEL[arch], zorder=3)
            if arch not in labels: handles.append(ln); labels.append(arch)
        ax.set_yscale('log')
        style_axes(ax, ylabel='Training cross-entropy (log)',
                   xlabel='Optimizer step', title=TASK_LABEL[task])
    legend(fig, handles, [LABEL[a] for a in labels], ncol=7)
    fig.suptitle('Unsmoothed global-batch training cross-entropy, mean over three seeds',
                 fontsize=12, color=INK, y=TITLE_Y)
    fig.tight_layout(rect=[0, 0, 1, RECT_TOP])
    fig.savefig(f'{FIG}/{out}', dpi=150, facecolor=SURFACE)
    plt.close(fig); print('wrote', out)


if __name__ == '__main__':
    runs = load()
    print(f'{len(runs)} runs loaded\n')
    print(table1(runs))
    fig2(runs); fig15(runs)
    for t in ['parity', 's5_swaps']:
        if any(k[0] == t for k in runs):
            fig3(runs, t, out=f'fig3_{t}_checkpoints.png')
            fig13(runs, t, out=f'fig13_{t}_per_seed.png')
