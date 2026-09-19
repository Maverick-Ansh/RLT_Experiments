"""Shared plotting vocabulary for every figure in this repo.

Palette: the six RLT variants take categorical slots 1-6 IN FIXED ORDER; the decoder-only
Transformer is drawn as a neutral dashed reference rather than a seventh hue, which is also what
the paper does.  Validated (light surface #fcfcfb, adjacent pairlist):
    lightness band PASS | chroma floor PASS | CVD separation PASS (worst adjacent dE 9.1, protan)
    normal-vision floor PASS (worst adjacent dE 19.6) | contrast WARN -> relieved by shipping
    Table 1 / Table 2 beside every figure, and by a distinct marker per series.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURFACE   = '#fcfcfb'
INK       = '#0b0b0b'
INK_2     = '#52514e'
INK_MUTED = '#8a8984'
GRID      = '#e6e5e1'

# categorical slots 1-6, fixed order, never cycled
ORDER = ['rlt1_4+4', 'rlt1_5+3', 'rlt1_6+2', 'rlt1_7+1', 'rlt1_8+0', 'rlt0_4+4', 'gpt_8']
COLOR = {'rlt1_4+4': '#2a78d6', 'rlt1_5+3': '#eb6834', 'rlt1_6+2': '#1baf7a',
         'rlt1_7+1': '#eda100', 'rlt1_8+0': '#e87ba4', 'rlt0_4+4': '#008300',
         'gpt_8': INK}
# secondary encoding so identity never rests on colour alone
MARKER = {'rlt1_4+4': 'o', 'rlt1_5+3': 's', 'rlt1_6+2': '^', 'rlt1_7+1': 'D',
          'rlt1_8+0': 'v', 'rlt0_4+4': 'P', 'gpt_8': 'x'}
DASH   = {a: '-' for a in ORDER}; DASH['gpt_8'] = '--'; DASH['rlt0_4+4'] = (0, (4, 2))
LABEL  = {'rlt1_4+4': 'RLT-1 4+4', 'rlt1_5+3': 'RLT-1 5+3', 'rlt1_6+2': 'RLT-1 6+2',
          'rlt1_7+1': 'RLT-1 7+1', 'rlt1_8+0': 'RLT-1 8+0',
          'rlt0_4+4': 'RLT-0 4+4', 'gpt_8': 'Transformer 8'}
TASK_LABEL = {'addition': 'Addition (1-8 digits; teacher forced)', 'parity': 'Parity',
              'mod5_flat': 'Mod 5: no brackets', 'mod5_brackets': 'Mod 5: with brackets',
              's5_swaps': r'$S_5$: swaps', 's5_standard': r'$S_5$: standard'}

# Reserved vertical space: suptitle sits at TITLE_Y, the single-row legend at LEGEND_Y, and axes
# are laid out below RECT_TOP.  Keeping these in one place is what stops the legend from landing
# on the title when the number of series changes.
TITLE_Y, LEGEND_Y, RECT_TOP = 0.995, 0.958, 0.928


def style_axes(ax, ylabel=None, xlabel=None, title=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, width=0.8)
    if ylabel: ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    if xlabel: ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    if title:  ax.set_title(title, color=INK, fontsize=10, pad=6)


def new_fig(*a, **kw):
    fig, ax = plt.subplots(*a, **kw)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def legend(fig, handles, labels, ncol=7, y=None):
    return fig.legend(handles, labels, loc='upper center',
                      bbox_to_anchor=(0.5, LEGEND_Y if y is None else y),
                      ncol=ncol, frameon=False, fontsize=8.5, handlelength=2.2,
                      columnspacing=1.2, handletextpad=0.5, labelcolor=INK_2)


def reference_line(ax, y, text=None):
    """Uniform-prediction / majority-class floor.  Recessive, never a series colour."""
    ax.axhline(y, color=INK_MUTED, linewidth=0.9, linestyle=':', zorder=1)
    if text:
        ax.annotate(text, xy=(0.99, y), xycoords=('axes fraction', 'data'),
                    ha='right', va='bottom', fontsize=7, color=INK_MUTED)
