"""Correctness gates for RLT-1.  float64 on CPU so the tolerances mean something.

  A  Prop. B.1  invariance to the prompt/response split
  B  Prop. G.1  causality (perturbation + autograd)
  C  Sec. 2.5   the SWA window rule (W includes the current token; W-1 are retained)
  D  App. G/H   gradient paths through s_t vs through decoder KV
  E  App. I.1   RLT-2 with B=1 equals RLT-1, checked against a NAIVE reference implementation
  F  App. B.3   the recurrent path traverses t*L_D decoder blocks
"""
import sys, itertools
import torch
sys.path.insert(0, '/content/RLT_Experiments')
torch.set_default_dtype(torch.float64)

from rlt.config import RLTConfig
from rlt.model import RLT
from rlt.layers import prefix_mask

fails = []
def check(name, cond, extra=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {extra}")
    if not cond: fails.append(name)

def tiny(**kw):
    base = dict(vocab_size=23, d_model=32, n_heads=4, d_ff=48, L_enc=2, L_dec=2,
                window=4, mup_base_width=32)
    base.update(kw)
    torch.manual_seed(7)
    m = RLT(RLTConfig(**base)).double().eval()
    # s_star and the gate bias are zero-initialized for training; leaving them at zero would make
    # several of these gates vacuously pass, so perturb them.
    with torch.no_grad():
        if m.s_star is not None:
            m.s_star.normal_(0, 0.5); m.merge.w_g.bias.normal_(0, 0.5)
    return m

B, T = 3, 11
torch.manual_seed(0)
X = torch.randint(1, 23, (B, T))

# ---------------------------------------------------------------------------
print('A. Prop. B.1 -- invariance to the serving split')
m = tiny()
full = m(X)                                                     # prefill all T at once
for split in [1, 4, 7, T - 1]:
    _, st = m(X[:, :split], return_state=True)
    logits = None
    for j in range(split, T):
        logits, st = m.step(X[:, j], st)
    d_log = (logits[:, 0] - full[:, T - 1]).abs().max().item()
    p1 = torch.softmax(logits[:, 0], -1); p2 = torch.softmax(full[:, T - 1], -1)
    check(f'split at {split:>2}: next-token logits agree', d_log < 1e-10, f'max|d|={d_log:.2e}')
    check(f'split at {split:>2}: distributions agree',
          (p1 - p2).abs().max().item() < 1e-10, f'max|dp|={(p1-p2).abs().max().item():.2e}')
_, st_a = m(X, return_state=True)
_, st_b = m(X[:, :5], return_state=True)
for j in range(5, T):
    _, st_b = m.step(X[:, j], st_b)
check('final recurrent state s_T identical',
      (st_a['s'] - st_b['s']).abs().max().item() < 1e-10,
      f"max|ds|={(st_a['s']-st_b['s']).abs().max().item():.2e}")
check('every decoder SWA cache identical',
      all((a[0] - b[0]).abs().max().item() < 1e-10 and (a[1] - b[1]).abs().max().item() < 1e-10
          for a, b in zip(st_a['caches'], st_b['caches'])))

# ---------------------------------------------------------------------------
print('\nB. Prop. G.1 -- causality')
m = tiny()
base = m(X)
worst = 0.0
for j in range(1, T):
    Xp = X.clone(); Xp[:, j] = (Xp[:, j] % 21) + 1              # change token j
    pert = m(Xp)
    worst = max(worst, (pert[:, :j] - base[:, :j]).abs().max().item())
check('perturbing token j leaves all logits < j bit-identical', worst == 0.0, f'max|d|={worst:.3e}')
# the perturbation must actually do something, or the test is vacuous
moved = min((m(torch.cat([X[:, :j], ((X[:, j:j+1] % 21) + 1), X[:, j+1:]], 1))[:, j:]
             - base[:, j:]).abs().max().item() for j in range(1, T))
check('and it DOES change logits >= j (test is not vacuous)', moved > 1e-6, f'min move={moved:.2e}')

# autograd version: d loss_t / d e_j must be exactly zero for j > t
m = tiny()
e, mk, mv, _, pos = m.encode(X)
e = e.detach().requires_grad_(True)
y, s, _ = m.run_decoder(e, mk, mv, pos)
t = 5
m.readout(m.norm_o(y[:, t])).sum().backward()
fut = e.grad[:, t + 1:].abs().max().item()
past = e.grad[:, :t + 1].abs().max().item()
check('autograd: d(logits_t)/d(e_j) == 0 for j > t', fut == 0.0, f'{fut:.3e}')
check('autograd: d(logits_t)/d(e_j) != 0 for j <= t', past > 1e-9, f'{past:.3e}')

# ---------------------------------------------------------------------------
print('\nC. Sec. 2.5 -- the sliding-window rule')
for W in [1, 2, 4, 8]:
    m = tiny(window=W)
    _, st = m(X, return_state=True)
    k = st['caches'][0][0]
    n = 0 if k is None else k.shape[1]
    check(f'W={W}: cache retains min(T, W-1) = {min(T, W-1)} entries', n == min(T, W - 1), f'got {n}')
check('W=1 retains nothing (paper: "this set is empty for W = 1")',
      tiny(window=1)(X, return_state=True)[1]['caches'][0][0] is None)

# ---------------------------------------------------------------------------
print('\nD. App. G/H -- gradient paths through the two halves of H_t = (s_t, C^D_t)')
# The sharp test: s_star reaches the loss ONLY through the recurrence.  Put a TBPTT boundary at
# t=BND and ask what survives.  (Gradient NORM is the wrong instrument here -- removing a path can
# increase it, since contributions cancel.  An exact zero cannot lie.)
BND = 5
def grad_s_star(detach):
    torch.manual_seed(7)
    m = tiny()
    e, mk, mv, _, pos = m.encode(X)
    s = m.s_star.expand(B, 1, m.cfg.d_model)
    caches = [(None, None)] * m.cfg.L_dec
    outs = []
    for t in range(T):
        u = m.merge(e[:, t:t+1], s)
        z, caches = m.decoder(u, pos[t:t+1], caches, mk, mv, m.rope, m.scale,
                              prefix_mask(pos[t:t+1], pos), None)
        s = z[:, -1:]
        outs.append(z)
        if t == BND:                                    # App. H, Eq. (H.1)
            if 's' in detach:   s = s.detach()
            if 'kv' in detach:  caches = [(a.detach() if a is not None else None,
                                           b.detach() if b is not None else None)
                                          for a, b in caches]
    m.readout(m.norm_o(torch.cat(outs, 1)[:, -1])).sum().backward()
    return m.s_star.grad.abs().max().item()

g_full  = grad_s_star([])
g_s     = grad_s_star(['s'])
g_kv    = grad_s_star(['kv'])
g_both  = grad_s_star(['s', 'kv'])
print(f'   |d loss_T / d s_star| : full={g_full:.3e}  detach s={g_s:.3e}  '
      f'detach KV={g_kv:.3e}  detach both={g_both:.3e}')
check('full BPTT: s_star reaches a loss 5 tokens past the boundary', g_full > 1e-12)
check('detaching s_t ALONE does not cut it -- the path survives through decoder KV',
      g_s > 1e-12, f'{g_s:.3e}')
check('detaching decoder KV ALONE does not cut it -- the path survives through s_t',
      g_kv > 1e-12, f'{g_kv:.3e}')
check('detaching BOTH cuts the recurrence EXACTLY (Eq. H.1)', g_both == 0.0, f'{g_both:.3e}')
check('the two partial detaches differ from full BPTT and from each other',
      g_s != g_full and g_kv != g_full and g_s != g_kv)

# ---------------------------------------------------------------------------
print('\nE. App. I.1 -- RLT-2 with B=1 equals RLT-1, vs a NAIVE reference')
def naive_rlt1(m, X):
    """Deliberately dumb reference: keep the entire decoder KV history in python lists and rebuild
    the window from scratch at every step.  No rolling cache, no reused masks."""
    from rlt.layers import attend
    cfg = m.cfg
    e, mk, mv, _, pos = m.encode(X)
    s = m.s_star.expand(X.shape[0], 1, cfg.d_model)
    hist_k = [[] for _ in range(cfg.L_dec)]
    hist_v = [[] for _ in range(cfg.L_dec)]
    outs = []
    for t in range(X.shape[1]):
        z = m.merge(e[:, t:t+1], s)
        for l, layer in enumerate(m.decoder.layers):
            h = layer.norm_S(z)
            q, k, v = layer.qkv(h).chunk(3, -1)
            Bn = z.shape[0]; H, Dh = cfg.n_heads, cfg.head_dim
            q = m.rope(q.view(Bn,1,H,Dh), pos[t:t+1]).reshape(Bn,1,-1)
            k = m.rope(k.view(Bn,1,H,Dh), pos[t:t+1]).reshape(Bn,1,-1)
            hist_k[l].append(k); hist_v[l].append(v)
            lo = max(0, t - cfg.window + 1)                      # window INCLUDES t
            K = torch.cat(hist_k[l][lo:t+1], 1); V = torch.cat(hist_v[l][lo:t+1], 1)
            z = z + layer.o_proj(attend(q, K, V, H, m.scale, None))
            qm = m.rope(layer.q_m(layer.norm_M(z)).view(Bn,1,H,Dh), pos[t:t+1]).reshape(Bn,1,-1)
            mmask = (pos[None, :] <= pos[t:t+1][:, None])
            z = z + layer.o_m(attend(qm, mk[0], mv[0], H, m.scale, mmask))
            z = z + layer.ffn(layer.norm_D(z))
        s = z
        outs.append(z)
    return m.readout(m.norm_o(torch.cat(outs, 1)))

for W in [2, 4, 8, 32]:
    m = tiny(window=W, chunk=1)
    d = (m(X) - naive_rlt1(m, X)).abs().max().item()
    check(f'W={W}: chunked B=1 == naive token-at-a-time reference', d < 1e-11, f'max|d|={d:.2e}')

print('   and RLT-2 at B>1 is a genuinely different model:')
m1 = tiny(chunk=1); base = m1(X)
for Bc in [2, 4, 8, 64]:
    m2 = tiny(chunk=Bc)
    m2.load_state_dict(m1.state_dict())
    o = m2(X)
    d0 = (o[:, 0] - base[:, 0]).abs().max().item()
    d1 = (o[:, 1:] - base[:, 1:]).abs().max().item()
    check(f'B={Bc:>2}: position 0 identical to RLT-1', d0 < 1e-11, f'{d0:.1e}')
    check(f'B={Bc:>2}: later positions differ (state is held across the chunk)',
          d1 > 1e-8, f'{d1:.1e}')

# ---------------------------------------------------------------------------
print('\nF. App. B.3 -- recurrent path length')
m = tiny()
e, mk, mv, _, pos = m.encode(X)
s0 = m.s_star.clone().requires_grad_(True)
s = s0.expand(B, 1, m.cfg.d_model)
caches = [(None, None)] * m.cfg.L_dec
for t in range(T):
    u = m.merge(e[:, t:t+1], s)
    z, caches = m.decoder(u, pos[t:t+1], caches, mk, mv, m.rope, m.scale,
                          prefix_mask(pos[t:t+1], pos), None)
    s = z
g = torch.autograd.grad(s.sum(), s0)[0].norm().item()
check(f'd s_T / d s_star is nonzero after T={T} tokens', g > 0, f'|grad|={g:.4e}')
print(f'   the path from s_star to s_T traverses T*L_D = {T}*{m.cfg.L_dec} = {T*m.cfg.L_dec} '
      f'decoder blocks, while each token evaluates L_E+L_D = {m.cfg.L_enc+m.cfg.L_dec}')

print('\n' + ('ALL GATES PASSED' if not fails else f'FAILURES: {fails}'))
sys.exit(1 if fails else 0)
