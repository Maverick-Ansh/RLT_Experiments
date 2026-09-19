"""The App. I.4 benchmark as a figure: what the chunk size B actually buys and costs.

Four panels, because App. I.4 asks for four different things:
  (a) training throughput vs B, against the RLT-0 and Transformer references
  (b) task accuracy vs B  -- the other half of the trade
  (c) prefill latency vs prompt length
  (d) time per generated token vs cached context length

B is ORDINAL, so it goes on the x-axis wherever possible rather than becoming eight hues.
Panels (c) and (d) need one series per variant, so they use categorical slots 1-5 in fixed order.
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, '/content/RLT_Experiments')
import matplotlib.pyplot as plt
from rlt.viz import (style_axes, legend, reference_line, SURFACE, INK, INK_2, INK_MUTED, GRID,
                     TITLE_Y, RECT_TOP)

R = '/content/RLT_Experiments/results'
FIG = '/content/RLT_Experiments/figures'
BS = [1, 2, 4, 8, 16, 32]
# categorical slots 1-5, fixed order (validated for the adjacent pairlist)
SER = [('RLT-2 B=1', '#2a78d6', 'o', '-'), ('RLT-2 B=8', '#eb6834', 's', '-'),
       ('RLT-2 B=32', '#1baf7a', '^', '-'), ('RLT-0 4+4', '#eda100', 'P', (0, (4, 2))),
       ('Transformer 8', INK, 'x', '--')]

cost = json.load(open(f'{R}/cost_bench.json'))
th = {r['variant']: r for r in cost['throughput']}

acc = {}
for f in glob.glob(f'{R}/rlt2/*.json'):
    r = json.load(open(f))
    B = r.get('chunk') or int(os.path.basename(f).split('__B')[1].split('__')[0])
    acc.setdefault(r['task'], {}).setdefault(B, []).append(r['final']['final'])
for f in glob.glob(f'{R}/main/*__rlt0_4+4__*.json'):
    r = json.load(open(f))
    acc.setdefault(r['task'], {}).setdefault('RLT-0', []).append(r['final']['final'])

fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))
fig.patch.set_facecolor(SURFACE)

# ---- (a) throughput --------------------------------------------------------
ax = axes[0, 0]
xs = np.arange(len(BS))
ys = [th[f'RLT-2 B={b}']['tokens_per_s'] for b in BS]
ax.plot(xs, ys, color='#2a78d6', linewidth=2, marker='o', markersize=6, zorder=3)
for name, c, ls in [('RLT-0 4+4', '#eda100', (0, (4, 2))), ('Transformer 8', INK, '--')]:
    if name in th:
        ax.axhline(th[name]['tokens_per_s'], color=c, linestyle=ls, linewidth=1.8, zorder=2)
        ax.annotate(name, xy=(0.02, th[name]['tokens_per_s']), xycoords=('axes fraction', 'data'),
                    fontsize=7.5, color=c, va='bottom')
ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in BS])
for i, (x, y) in enumerate(zip(xs, ys)):
    ax.annotate(f"{th[f'RLT-2 B={BS[i]}']['speedup_vs_B1']:.1f}x", (x, y), textcoords='offset points',
                xytext=(0, 7), ha='center', fontsize=7, color=INK_2)
style_axes(ax, ylabel='Training tokens/s (fwd+bwd+opt)', xlabel='Chunk size B',
           title='(a) What parallelism B buys')

# ---- (b) accuracy ----------------------------------------------------------
ax = axes[1, 0]
cols = {'parity': '#2a78d6', 's5_swaps': '#eb6834'}
mk = {'parity': 'o', 's5_swaps': 's'}
for task, per in sorted(acc.items()):
    bs = [b for b in BS if b in per]
    if not bs: continue
    mu = [np.mean(per[b]) for b in bs]
    sd = [np.std(per[b], ddof=1) if len(per[b]) > 1 else 0 for b in bs]
    ax.errorbar([BS.index(b) for b in bs], mu, yerr=sd, color=cols.get(task, '#1baf7a'),
                linewidth=2, marker=mk.get(task, '^'), markersize=6, capsize=3, label=task, zorder=3)
    if 'RLT-0' in per:
        ax.axhline(np.mean(per['RLT-0']), color=cols.get(task, '#1baf7a'), linestyle=(0, (4, 2)),
                   linewidth=1.4, alpha=0.8, zorder=2)
        ax.annotate(f'{task} RLT-0', xy=(0.98, np.mean(per['RLT-0'])),
                    xycoords=('axes fraction', 'data'), ha='right', va='bottom',
                    fontsize=7, color=cols.get(task, '#1baf7a'))
ax.set_xticks(np.arange(len(BS))); ax.set_xticklabels([str(b) for b in BS])
ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
style_axes(ax, ylabel='Final-answer accuracy (%)', xlabel='Chunk size B',
           title='(b) What it costs  ·  dashed = RLT-0 (no feedback at all)')

# ---- (c) prefill latency ---------------------------------------------------
ax = axes[0, 1]
pre = {r['variant']: r for r in cost['prefill_latency_s']}
Ls = [32, 64, 128, 256]
for name, c, m, ls in SER:
    if name not in pre: continue
    ax.plot(np.arange(len(Ls)), [pre[name][str(L)] * 1e3 for L in Ls], color=c, marker=m,
            markersize=5, linewidth=2, linestyle=ls, label=name, zorder=3)
ax.set_xticks(np.arange(len(Ls))); ax.set_xticklabels([str(L) for L in Ls])
ax.set_yscale('log')
style_axes(ax, ylabel='Prefill latency (ms, log)', xlabel='Prompt length (tokens)',
           title='(c) Prefill: T sequential decoder updates')

# ---- (d) per-token generation ---------------------------------------------
ax = axes[1, 1]
gen = {r['variant']: r for r in cost['per_token_gen_s']}
Cs = [32, 128, 256]
for name, c, m, ls in SER:
    if name not in gen: continue
    ax.plot(np.arange(len(Cs)), [gen[name][str(L)] * 1e3 for L in Cs], color=c, marker=m,
            markersize=5, linewidth=2, linestyle=ls, label=name, zorder=3)
ax.set_xticks(np.arange(len(Cs))); ax.set_xticklabels([str(L) for L in Cs])
style_axes(ax, ylabel='Time per generated token (ms)', xlabel='Cached context length',
           title='(d) Generation is one token at a time for everyone')

h, l = axes[0, 1].get_legend_handles_labels()
legend(fig, h, l, ncol=5)
fig.suptitle("App. I.4's matched benchmark, which the paper specifies but does not run",
             fontsize=12, color=INK, y=TITLE_Y)
fig.text(0.5, 0.012, f"{cost['hardware']} · fp32 · batch {cost['batch']} (a,b) / 32 (c,d) · "
         f"width {cost['d_model']} · warmup and device synchronization on every measurement",
         ha='center', fontsize=8, color=INK_MUTED)
fig.tight_layout(rect=[0, 0.03, 1, RECT_TOP])
fig.savefig(f'{FIG}/fig_rlt2_benchmark.png', dpi=150, facecolor=SURFACE)
plt.close(fig)
print('wrote fig_rlt2_benchmark.png')
