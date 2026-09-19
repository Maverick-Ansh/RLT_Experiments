"""App. D.3 / D.4 / G / H: exact current-policy replay on a recurrent decoder.

float64 on CPU so "exactly 1" means exactly 1.
"""
import sys, json
from contextlib import nullcontext
import torch
import torch.nn.functional as F
sys.path.insert(0, '/content/RLT_Experiments')
torch.set_default_dtype(torch.float64)
from rlt.config import RLTConfig
from rlt.model import RLT

fails, OUT = [], {}
def check(name, cond, extra=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {extra}")
    if not cond: fails.append(name)

torch.manual_seed(11)
cfg = RLTConfig(vocab_size=23, d_model=32, n_heads=4, d_ff=48, L_enc=2, L_dec=2,
                window=4, mup_base_width=32)
model = RLT(cfg).double()
with torch.no_grad():
    model.s_star.normal_(0, .5); model.merge.w_g.bias.normal_(0, .5)

B, Tp, N = 4, 7, 6                      # batch, prompt length, response length
c = torch.randint(1, 23, (B, Tp))


# ---------------------------------------------------------------------------
# SAMPLER: process the full prompt, sample a response, record log mu for each action.
# App. F.5 step 1: "Behaviour probabilities include all sampling transforms and renormalization."
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample(model, c, N, temperature=1.0, top_k=None):
    logits, st = model(c, return_state=True)
    ys, logmu, violations = [], [], 0
    nxt_logits = logits[:, -1]
    g = torch.Generator().manual_seed(3)
    for _ in range(N):
        z = nxt_logits / temperature
        full = F.log_softmax(nxt_logits, -1)              # the UNTRUNCATED target policy
        if top_k is not None:
            v, _ = z.topk(top_k, dim=-1)
            z = z.masked_fill(z < v[:, -1:], float('-inf'))
        lp = F.log_softmax(z, -1)                         # the BEHAVIOUR policy mu
        violations += int((torch.isinf(lp) & torch.isfinite(full)).sum())
        y = torch.multinomial(lp.exp(), 1, generator=g)[:, 0]
        ys.append(y); logmu.append(lp.gather(1, y[:, None])[:, 0])
        nxt_logits, st = model.step(y, st)
        nxt_logits = nxt_logits[:, 0]
    return torch.stack(ys, 1), torch.stack(logmu, 1), violations


y, logmu, _ = sample(model, c, N)


# ---------------------------------------------------------------------------
# TRAINER (App. F.5 steps 2-4): rebuild encoder features, recurrent outputs and every decoder SWA
# cache under the CURRENT parameters, from H_0 = (s_star, empty), then read each sampled action's
# log-probability off the state that precedes it.
#
# The gradient boundary sits at position P-1, the last prompt token -- that is the state which
# produces the first action's logits, so everything from there on must stay differentiable.
# ---------------------------------------------------------------------------
def replay(model, c, y, boundary_no_grad=False, detach=()):
    full = torch.cat([c, y], 1)
    P = c.shape[1]
    bnd = P - 1
    e, mk, mv, _, pos = model.encode(full)
    if 'enc_mem' in detach:
        mk = [t.detach() for t in mk]; mv = [t.detach() for t in mv]
    # App. D.4 / F.5: "Prompt replay under no_grad, or detaching prompt KV, can keep forward values
    # unchanged while dropping gradients required by full BPTT."
    with (torch.no_grad() if boundary_no_grad else nullcontext()):
        _, s, caches = model.run_decoder(e[:, :bnd], mk, mv, pos[:bnd], mem_pos=pos)
    if 's' in detach:  s = s.detach()
    if 'kv' in detach: caches = [(a.detach() if a is not None else None,
                                  b.detach() if b is not None else None) for a, b in caches]
    hid, _, _ = model.run_decoder(e[:, bnd:], mk, mv, pos[bnd:],
                                  s=s, caches=caches, mem_pos=pos)
    logits = model.readout(model.norm_o(hid))      # positions bnd .. P+N-1
    lp = F.log_softmax(logits[:, :N], -1)          # position P-1+i predicts y_i
    return lp.gather(2, y[:, :, None])[:, :, 0]


def flat_grad(model):
    """Zeros for parameters that receive NO gradient, so the vectors stay comparable."""
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).clone()
                      for p in model.parameters()])


print('1. replay at UNCHANGED parameters must reproduce the sampler exactly')
lp = replay(model, c, y)
r = (lp - logmu).exp()
check('log-ratio is exactly zero', (lp - logmu).abs().max().item() < 1e-12,
      f'max|log r| = {(lp-logmu).abs().max().item():.3e}')
check('importance ratio is exactly 1', (r - 1).abs().max().item() < 1e-12,
      f'max|r-1| = {(r-1).abs().max().item():.3e}')
OUT['exact_replay_max_log_ratio'] = (lp - logmu).abs().max().item()

print('\n2. after a parameter update the ratio moves, and the score-function gradient exists')
with torch.no_grad():
    for p in model.parameters(): p.add_(torch.randn_like(p) * 1e-3)
lp2 = replay(model, c, y)
r2 = (lp2 - logmu).exp()
check('ratio is no longer 1', (r2 - 1).abs().max().item() > 1e-6,
      f'max|r-1| = {(r2-1).abs().max().item():.3e}')
adv = torch.randn(B)                                     # R(c,y) - b(c), held constant
model.zero_grad(); (-(adv[:, None] * lp2).sum()).backward()
gn = flat_grad(model).norm().item()
check('policy gradient reaches every parameter',
      all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()),
      f'|grad| = {gn:.4f}')

print('\n3. App. D.4 -- the no_grad prompt trap: identical forward, different gradient')
def grad_of(**kw):
    model.zero_grad()
    lp_ = replay(model, c, y, **kw)
    (-(adv[:, None] * lp_).sum()).backward()
    return lp_.detach(), flat_grad(model)

lp_full, g_full = grad_of()
lp_ng,   g_ng   = grad_of(boundary_no_grad=True)
cos = F.cosine_similarity(g_full[None], g_ng[None]).item()
rel = ((g_ng - g_full).norm() / g_full.norm()).item()
check('forward log-probs are IDENTICAL', (lp_full - lp_ng).abs().max().item() < 1e-12,
      f'max|dlogp| = {(lp_full-lp_ng).abs().max().item():.3e}')
check('...but the gradient is a different vector', rel > 1e-6,
      f'relative L2 diff = {rel:.4f},  cosine = {cos:.6f}')
model.zero_grad(); replay(model, c, y, boundary_no_grad=True).sum().backward()
check('and s_star receives NO gradient at all under no_grad prompt replay',
      model.s_star.grad is None or model.s_star.grad.abs().max().item() == 0.0)
OUT['no_grad_prompt'] = dict(cosine=cos, rel_l2=rel,
                             forward_max_diff=(lp_full-lp_ng).abs().max().item())

print('\n4. Eq. (G.4)-(G.5) -- detaching boundary tensors drops the matching gradient terms')
rows = []
for det in [('s',), ('kv',), ('enc_mem',), ('s', 'kv'), ('s', 'kv', 'enc_mem')]:
    lp_d, g_d = grad_of(detach=det)
    co = F.cosine_similarity(g_d[None], g_full[None]).item()
    rl = ((g_d - g_full).norm() / g_full.norm()).item()
    fwd = (lp_d - lp_full).abs().max().item()
    rows.append(dict(detach='+'.join(det), cosine=co, rel_l2=rl, forward_max_diff=fwd))
    print(f"   detach {'+'.join(det):<16} forward max|d|={fwd:.1e}   "
          f'cosine={co:.6f}   relative L2={rl:.4f}')
check('every detach leaves the forward pass bit-identical',
      all(v['forward_max_diff'] < 1e-12 for v in rows))
check('every detach changes the gradient', all(v['rel_l2'] > 1e-9 for v in rows))
check('detaching more removes more',
      rows[-1]['rel_l2'] >= max(rows[0]['rel_l2'], rows[1]['rel_l2'], rows[2]['rel_l2']))
i_sk = [v['detach'] for v in rows].index('s+kv')
check('no_grad prompt == detaching (s, KV) at the same boundary',
      abs(rows[i_sk]['rel_l2'] - rel) < 1e-9,
      f"{rows[i_sk]['rel_l2']:.6f} vs {rel:.6f}")
OUT['detach_rows'] = rows

print('\n5. App. D.3 support condition under truncated sampling')
for k in [None, 8, 3, 1]:
    _, _, viol = sample(model, c, N, top_k=k)
    lbl = 'untruncated' if k is None else f'top-k={k}'
    print(f'   {lbl:<14} actions with mu=0 but p_Theta>0 : {viol}')
    if k is None:
        check('untruncated sampling satisfies p << mu', viol == 0)
    else:
        check(f'{lbl} VIOLATES the support condition (as the paper warns)', viol > 0, f'{viol}')
    OUT[f'support_violations_{lbl}'] = viol

json.dump(OUT, open('/content/RLT_Experiments/results/rl_replay.json', 'w'), indent=1)
print('\n' + ('ALL RL REPLAY CHECKS PASSED' if not fails else f'FAILURES: {fails}'))
sys.exit(1 if fails else 0)
