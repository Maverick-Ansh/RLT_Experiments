"""Length generalization from the best in-distribution checkpoint.  Sec 3.5 / App. J.5.

Checkpoint selection reads ONLY the validation loss recorded during training.  App. J.5:
"Test results do not participate in selection."
"""
import json, math, os
import torch

from .build import build
from .train import metrics
from . import data as D
from . import tasks as T


def load_ckpt(path, device='cuda'):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build(ck['arch'], T.VOCAB, d_model=ck['d_model']).to(device)
    m.load_state_dict(ck['state'])
    m.eval()
    return m, ck


@torch.no_grad()
def score_length(model, task, length, n=256, device='cuda', micro=None):
    x, y, msk = D.test_set(task, n, length, device)
    if micro is None:                     # long S5 sequences need smaller forward batches
        micro = 256 if x.shape[1] <= 128 else (64 if x.shape[1] <= 300 else 32)
    agg = dict(c_tok=0, n_tok=0, c_fin=0, c_seq=0, n_ex=0)
    for i in range(0, x.size(0), micro):
        r = metrics(model(x[i:i+micro]), y[i:i+micro], msk[i:i+micro])
        agg['c_tok'] += r['token']/100*r['n_tok']; agg['n_tok'] += r['n_tok']
        agg['c_fin'] += r['final']/100*r['n_ex']
        agg['c_seq'] += r['sequence']/100*r['n_ex']; agg['n_ex'] += r['n_ex']
    ne = max(agg['n_ex'], 1)
    return dict(length=length, n=ne, seq_len=int(x.shape[1]),
                token=100*agg['c_tok']/max(agg['n_tok'],1),
                final=100*agg['c_fin']/ne, sequence=100*agg['c_seq']/ne)


# ---------------------------------------------------------------------------
# Greedy generation -- needed only for App. J.7's "exact answer + EOS" metric, which is far
# stricter than teacher forcing: every answer token must be right AND the model must stop.
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_answer(model, x_prompt, max_new, device='cuda'):
    """x_prompt: (B, P) ending at '='.  Returns (B, max_new) generated tokens."""
    from .model import RLT
    out = []
    if isinstance(model, RLT):
        logits, st = model(x_prompt, return_state=True)
        nxt = logits[:, -1].argmax(-1)
        for _ in range(max_new):
            out.append(nxt)
            logits, st = model.step(nxt, st)
            nxt = logits[:, 0].argmax(-1)
    else:
        logits, caches = model(x_prompt, return_state=True)
        nxt = logits[:, -1].argmax(-1)
        t = x_prompt.shape[1]
        for _ in range(max_new):
            out.append(nxt)
            pos = torch.arange(t, t + 1, device=device)
            logits, caches = model(nxt[:, None], caches=caches, pos=pos, return_state=True)
            nxt = logits[:, 0].argmax(-1); t += 1
    return torch.stack(out, 1)


@torch.no_grad()
def exact_answer_acc(model, digits, n=256, device='cuda', micro=64):
    """App. J.7: greedy exact-answer accuracy -- the correct answer followed by EOS."""
    x, y, msk = D.test_set('addition', n, digits, device)
    correct = 0
    for i in range(0, x.size(0), micro):
        xb, yb, mb = x[i:i+micro], y[i:i+micro], msk[i:i+micro]
        eq = (xb == T.EQ).float().argmax(1)                         # index of '='
        P = int(eq.max().item()) + 1
        assert (eq == eq[0]).all(), 'fixed-width test set => aligned prompts'
        gold = yb * mb                                              # answer tokens incl. EOS
        n_ans = int(mb.sum(1).max().item())
        gen = greedy_answer(model, xb[:, :P], n_ans, device)
        tgt = torch.stack([yb[r, mb[r]] for r in range(xb.shape[0])])
        correct += (gen[:, :tgt.shape[1]] == tgt).all(1).sum().item()
    return 100.0 * correct / x.size(0)


def wilson(k, n, z=1.959963985):
    """Pointwise 95% Wilson interval (App. J.7 reports these for exact-answer accuracy)."""
    if n == 0: return (0.0, 0.0)
    p = k / n
    den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return (100*max(0.0, c-h), 100*min(1.0, c+h))
