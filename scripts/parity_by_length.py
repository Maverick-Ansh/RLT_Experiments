"""Where does parity competence actually end?

The paper validates parity only at lengths 32/36/40 -- the hardest end of the 3-40 training
range.  A model that has genuinely learned parity for short strings but not long ones is
indistinguishable, under that protocol, from a model that learned nothing.  This script scores a
trained checkpoint at EVERY length in the training range, which separates the two.

Runs on CPU so it never contends with the sweep.
"""
import sys, glob, os, json
import numpy as np, torch
sys.path.insert(0, '/content/RLT_Experiments')
from rlt.build import build
from rlt import tasks as T

LENS = [3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40]
N = 256


@torch.no_grad()
def score(model, L, rng):
    x, y, m = T.gen_parity(N, rng, length=L)
    x, y, m = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)
    pred = model(x).argmax(-1)
    return 100.0 * ((pred == y) & m).sum().item() / m.sum().item()


def main(pattern='/content/RLT_Experiments/results/fig3/parity__*__42.pt'):
    cks = sorted(glob.glob(pattern))
    if not cks:
        print('no checkpoints yet at', pattern); return
    print(f"{'arch':<12}{'step':>6}" + ''.join(f'{L:>6}' for L in LENS))
    print(' ' * 18 + ''.join(f'{"":>6}' for _ in LENS))
    out = {}
    for c in cks:
        ck = torch.load(c, map_location='cpu', weights_only=False)
        model = build(ck['arch'], T.VOCAB, d_model=ck['d_model'],
                      d_ff=ck['cfg'].get('d_ff')).eval()
        model.load_state_dict(ck['state'])
        rng = np.random.default_rng(12345)          # same examples for every checkpoint
        accs = [score(model, L, rng) for L in LENS]
        out[ck['arch']] = accs
        print(f"{ck['arch']:<12}{ck['step']:>6}" + ''.join(f'{a:>6.1f}' for a in accs))
    json.dump({'lengths': LENS, 'acc': out},
              open('/content/RLT_Experiments/results/parity_by_length.json', 'w'), indent=1)
    print('\nuniform floor 50.0%   (paper validates only at 32/36/40)')


if __name__ == '__main__':
    main(*sys.argv[1:])
