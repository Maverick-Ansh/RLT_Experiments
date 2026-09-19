"""The figure the paper's protocol cannot draw: parity accuracy resolved by sequence length.

The paper scores parity only at lengths 32/36/40, so a model that solves parity up to length 12
and one that never learned anything both report ~50%.  Scoring every length separates them and
recovers the depth-split ordering that the pooled metric destroys.
"""
import sys, glob, json, os
import numpy as np, torch
sys.path.insert(0, '/content/RLT_Experiments')
from rlt.build import build
from rlt import tasks as T
from rlt import viz
import matplotlib.pyplot as plt

LENS = [3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40]
N, SEEDS = 512, [42, 43, 44]
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RES = '/content/RLT_Experiments/results/fig3'
OUT = '/content/RLT_Experiments/figures'


@torch.no_grad()
def score(model, L):
    rng = np.random.default_rng(12345 + L)          # identical examples for every model
    x, y, m = T.gen_parity(N, rng, length=L)
    x = torch.from_numpy(x).to(DEV); y = torch.from_numpy(y).to(DEV)
    mk = torch.from_numpy(m).to(DEV)
    pred = model(x).argmax(-1)
    return 100.0 * ((pred == y) & mk).sum().item() / mk.sum().item()


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {}
    for ck_path in sorted(glob.glob(f'{RES}/parity__*.pt')):
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        m = build(ck['arch'], T.VOCAB, d_model=ck['d_model'], d_ff=ck['cfg'].get('d_ff')).eval().to(DEV)
        m.load_state_dict(ck['state'])
        data.setdefault(ck['arch'], []).append([score(m, L) for L in LENS])
        print('scored', os.path.basename(ck_path), flush=True)
    if not data:
        print('no checkpoints'); return
    json.dump({'lengths': LENS, 'acc': {k: v for k, v in data.items()}},
              open('/content/RLT_Experiments/results/parity_by_length.json', 'w'), indent=1)

    archs = [a for a in viz.ORDER if a in data]
    fig, ax = viz.new_fig(figsize=(7.6, 4.6))
    handles, labels = [], []
    for a in archs:
        v = np.array(data[a])                        # (seeds, lengths)
        mu, sd = v.mean(0), (v.std(0, ddof=1) if len(v) > 1 else np.zeros(len(LENS)))
        ln, = ax.plot(LENS, mu, color=viz.COLOR[a], linestyle=viz.DASH[a], linewidth=2.0,
                      marker=viz.MARKER[a], markersize=5, zorder=3)
        ax.fill_between(LENS, np.clip(mu - sd, 0, 100), np.clip(mu + sd, 0, 100),
                        color=viz.COLOR[a], alpha=0.13, linewidth=0, zorder=2)
        handles.append(ln); labels.append(f'{viz.LABEL[a]}  (n={len(v)})')
    viz.style_axes(ax, ylabel='Parity accuracy (%)', xlabel='Sequence length (bits)')
    viz.reference_line(ax, 50.0, 'uniform 50%')
    ax.axvspan(31, 41, color=viz.INK_MUTED, alpha=0.09, zorder=1)
    ax.annotate('the only region\nthe paper scores', xy=(36, 96), ha='center', va='top',
                fontsize=7.5, color=viz.INK_MUTED)
    ax.set_ylim(40, 104); ax.set_xlim(2, 41)
    viz.legend(fig, handles, labels, ncol=min(4, len(archs)), y=0.945)
    fig.suptitle('Parity by length, width 512, step 500 - resolved vs pooled',
                 y=viz.TITLE_Y, fontsize=11, color=viz.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    p = f'{OUT}/fig_parity_by_length.png'
    fig.savefig(p, dpi=170, facecolor=viz.SURFACE)
    print('wrote', p)

    print(f"\n{'arch':<14}" + ''.join(f'{L:>6}' for L in LENS))
    for a in archs:
        mu = np.array(data[a]).mean(0)
        print(f'{viz.LABEL[a]:<14}' + ''.join(f'{x:>6.1f}' for x in mu))


if __name__ == '__main__':
    main()
