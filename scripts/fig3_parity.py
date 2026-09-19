"""Figure 3 of the paper, reproduced at the paper's own width 512.

Left  : validation final-answer accuracy vs optimizer step, mean +- sample SD over seeds 42/43/44.
Right : the step-500 snapshot the paper tabulates in its Fig. 3 left panel, with this
        reproduction's mean +- SD and individual seeds, drawn beside the paper's published values.

The paper's numbers are hard-coded here ONLY as a published reference to compare against; every
number attributed to this reproduction is read from results/fig3/*.json.
"""
import json, glob, os, sys
import numpy as np
sys.path.insert(0, '/content/RLT_Experiments')
from rlt import viz
import matplotlib.pyplot as plt

RES = '/content/RLT_Experiments/results/fig3'
OUT = '/content/RLT_Experiments/figures'
os.makedirs(OUT, exist_ok=True)

# paper Fig. 3, left panel (step 500, 256,000 examples/seed).  RLT-0 is not run by the paper.
PAPER = {'rlt1_4+4': (83.3, 28.9), 'rlt1_5+3': (83.5, 28.0), 'rlt1_6+2': (99.4, 1.0),
         'rlt1_7+1': (82.8, 29.8), 'rlt1_8+0': (48.7, 2.6), 'gpt_8': (48.5, 0.5)}
UNIFORM, MAJORITY = 50.0, 53.61


def load():
    runs = {}
    for f in sorted(glob.glob(f'{RES}/parity__*.json')):
        d = json.load(open(f))
        runs.setdefault(d['arch'], []).append(d)
    return runs


def curve(rs):
    """App. J.3 seed aggregation: pool per step, sample SD (denominator n-1), no imputation."""
    steps = [h['step'] for h in rs[0]['history']]
    n = min(len(r['history']) for r in rs)
    steps = steps[:n]
    acc = np.array([[h['final'] for h in r['history'][:n]] for r in rs])
    return np.array(steps), acc.mean(0), acc.std(0, ddof=1)


def main():
    runs = load()
    archs = [a for a in viz.ORDER if a in runs]
    if not archs:
        print('no results in', RES); return

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    fig.patch.set_facecolor(viz.SURFACE)
    axL, axR = axes

    handles, labels = [], []
    for a in archs:
        s, m, sd = curve(runs[a])
        ln, = axL.plot(s, m, color=viz.COLOR[a], linestyle=viz.DASH[a], linewidth=2.0,
                       marker=viz.MARKER[a], markersize=4.5, markevery=max(1, len(s)//8),
                       zorder=3, label=viz.LABEL[a])
        axL.fill_between(s, np.clip(m - sd, 0, 100), np.clip(m + sd, 0, 100),
                         color=viz.COLOR[a], alpha=0.13, linewidth=0, zorder=2)
        handles.append(ln); labels.append(viz.LABEL[a])
    viz.style_axes(axL, ylabel='Final-answer accuracy (%)',
                   xlabel='Optimizer step (global batch 512)',
                   title='Parity, width 512 - this reproduction')
    viz.reference_line(axL, UNIFORM, 'uniform 50%')
    viz.reference_line(axL, MAJORITY, 'majority 53.6%')
    axL.set_ylim(-3, 103)

    x = np.arange(len(archs))
    for i, a in enumerate(archs):
        v = np.array([r['final']['final'] for r in runs[a]])
        axR.errorbar(i - 0.13, v.mean(), yerr=v.std(ddof=1), fmt=viz.MARKER[a],
                     color=viz.COLOR[a], markersize=7, capsize=3, elinewidth=1.4, zorder=4)
        axR.scatter(np.full(len(v), i - 0.13), v, marker='x', s=22,
                    color=viz.COLOR[a], alpha=0.75, zorder=5)
        if a in PAPER:
            pm, ps = PAPER[a]
            axR.errorbar(i + 0.13, pm, yerr=ps, fmt='o', markerfacecolor='none',
                         color=viz.INK_MUTED, markersize=6, capsize=3, elinewidth=1.2, zorder=3)
    viz.style_axes(axR, ylabel='Final-answer accuracy (%) at step 500',
                   title='Step 500 - filled: this run   open grey: paper Fig. 3')
    viz.reference_line(axR, UNIFORM, 'uniform 50%')
    axR.set_xticks(x)
    axR.set_xticklabels([viz.LABEL[a].replace(' ', '\n', 1) for a in archs], fontsize=7.5)
    axR.set_ylim(-3, 130)

    viz.legend(fig, handles, labels, ncol=len(archs))
    fig.suptitle('Parity at the paper width (512), seeds 42/43/44', y=viz.TITLE_Y,
                 fontsize=11, color=viz.INK)
    fig.tight_layout(rect=(0, 0, 1, viz.RECT_TOP))
    p = f'{OUT}/fig3_parity.png'
    fig.savefig(p, dpi=170, facecolor=viz.SURFACE)
    print('wrote', p)

    print(f"\n{'arch':<14}{'this run (step 500)':>24}{'paper Fig. 3':>18}")
    for a in archs:
        v = np.array([r['final']['final'] for r in runs[a]])
        pa = f'{PAPER[a][0]:.1f} +- {PAPER[a][1]:.1f}' if a in PAPER else 'not run by paper'
        print(f'{viz.LABEL[a]:<14}{v.mean():>15.2f} +- {v.std(ddof=1):<5.2f}{pa:>18}')


if __name__ == '__main__':
    main()
