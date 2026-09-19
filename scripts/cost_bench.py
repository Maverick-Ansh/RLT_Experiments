"""App. I.4's matched benchmark, plus App. B.2/B.3 and App. C cost measurements.

Reports, all with device synchronization and warmup:
  1  training tokens/s (forward + backward + optimizer) vs chunk size B, RLT-0, Transformer
  2  peak device memory
  3  prefill latency vs prompt length
  4  time per generated token vs cached context length
  5  numerical agreement: max and RMS |dlogit| across chunkwise / partial-chunk / tokenwise
  6  full-BPTT parameter-gradient agreement between execution modes
  7  the recurrent path length of App. B.3 (counted, not estimated)
"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, '/content/RLT_Experiments')
from rlt.build import build
from rlt.config import RLTConfig
from rlt.model import RLT
from rlt.train import masked_loss
from rlt import tasks as T

ap = argparse.ArgumentParser()
ap.add_argument('--d_model', type=int, default=128)
ap.add_argument('--batch', type=int, default=512)
ap.add_argument('--seq', type=int, default=64)
ap.add_argument('--out', default='/content/RLT_Experiments/results/cost_bench.json')
A = ap.parse_args()
dev = 'cuda'
torch.backends.cudnn.benchmark = True
GPU = torch.cuda.get_device_name(0)
OUT = {'hardware': GPU, 'dtype': 'float32', 'batch': A.batch, 'seq': A.seq,
       'd_model': A.d_model, 'torch': torch.__version__}


def variants():
    v = [(f'RLT-2 B={B}', 'rlt1_4+4', {'chunk': B}) for B in [1, 2, 4, 8, 16, 32]]
    v.append(('RLT-0 4+4', 'rlt0_4+4', {}))
    v.append(('Transformer 8', 'gpt_8', {}))
    return v


def sync(): torch.cuda.synchronize()


# ---------------------------------------------------------------------------
print(f'== 1/2  training throughput and peak memory  ({GPU}, fp32, batch {A.batch}, '
      f'seq {A.seq}) ==')
print(f"{'variant':<16}{'s/step':>10}{'tok/s':>12}{'peak GB':>10}{'vs B=1':>9}")
rows, base = [], None
x = torch.randint(0, 130, (A.batch, A.seq), device=dev)
y = torch.randint(0, 130, (A.batch, A.seq), device=dev)
m = torch.ones_like(x, dtype=torch.bool)
for name, arch, kw in variants():
    torch.manual_seed(0)
    mdl = build(arch, T.VOCAB, d_model=A.d_model, **kw).to(dev)
    opt = torch.optim.AdamW(mdl.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    for i in range(5):
        if i == 1: sync(); t0 = time.time()
        opt.zero_grad(set_to_none=True)
        masked_loss(mdl(x), y, m).backward()
        opt.step()
    sync()
    dt = (time.time() - t0) / 4
    mem = torch.cuda.max_memory_allocated() / 1e9
    base = base or dt
    toks = A.batch * A.seq / dt
    rows.append(dict(variant=name, s_per_step=dt, tokens_per_s=toks, peak_gb=mem,
                     speedup_vs_B1=base / dt))
    print(f'{name:<16}{dt:>10.3f}{toks:>12,.0f}{mem:>10.2f}{base/dt:>8.2f}x')
    del mdl, opt; torch.cuda.empty_cache()
OUT['throughput'] = rows

# ---------------------------------------------------------------------------
print(f'\n== 3  prefill latency vs prompt length  (batch 32) ==')
print(f"{'variant':<16}" + ''.join(f'{L:>9}' for L in [32, 64, 128, 256]))
pre = []
xb = {L: torch.randint(0, 130, (32, L), device=dev) for L in [32, 64, 128, 256]}
for name, arch, kw in variants():
    torch.manual_seed(0)
    mdl = build(arch, T.VOCAB, d_model=A.d_model, **kw).to(dev).eval()
    r = {}
    with torch.no_grad():
        for L, xx in xb.items():
            for i in range(4):
                if i == 1: sync(); t0 = time.time()
                mdl(xx)
            sync(); r[L] = (time.time() - t0) / 3
    pre.append(dict(variant=name, **{str(k): v for k, v in r.items()}))
    print(f'{name:<16}' + ''.join(f'{r[L]*1e3:>8.1f}ms' for L in [32, 64, 128, 256]))
    del mdl; torch.cuda.empty_cache()
OUT['prefill_latency_s'] = pre

# ---------------------------------------------------------------------------
print(f'\n== 4  time per generated token vs cached context length  (batch 32) ==')
print(f"{'variant':<16}" + ''.join(f'{L:>10}' for L in [32, 128, 256]))
gen = []
for name, arch, kw in variants():
    torch.manual_seed(0)
    mdl = build(arch, T.VOCAB, d_model=A.d_model, **kw).to(dev).eval()
    r = {}
    with torch.no_grad():
        for L in [32, 128, 256]:
            xx = torch.randint(0, 130, (32, L), device=dev)
            if isinstance(mdl, RLT):
                _, st = mdl(xx, return_state=True)
                tok = torch.randint(0, 130, (32,), device=dev)
                for i in range(9):
                    if i == 1: sync(); t0 = time.time()
                    _, st = mdl.step(tok, st)
            else:
                _, ca = mdl(xx, return_state=True)
                tok = torch.randint(0, 130, (32, 1), device=dev)
                t = L
                for i in range(9):
                    if i == 1: sync(); t0 = time.time()
                    _, ca = mdl(tok, caches=ca,
                                pos=torch.arange(t, t+1, device=dev), return_state=True)
                    t += 1
            sync(); r[L] = (time.time() - t0) / 8
    gen.append(dict(variant=name, **{str(k): v for k, v in r.items()}))
    print(f'{name:<16}' + ''.join(f'{r[L]*1e3:>9.2f}ms' for L in [32, 128, 256]))
    del mdl; torch.cuda.empty_cache()
OUT['per_token_gen_s'] = gen

# ---------------------------------------------------------------------------
print('\n== 5  numerical agreement across execution modes (float64, CPU) ==')
torch.set_default_dtype(torch.float64)
cfg = dict(vocab_size=23, d_model=32, n_heads=4, d_ff=48, L_enc=2, L_dec=2,
           window=4, mup_base_width=32)
torch.manual_seed(7)
ref = RLT(RLTConfig(chunk=1, **cfg)).double().eval()
with torch.no_grad():
    ref.s_star.normal_(0, .5); ref.merge.w_g.bias.normal_(0, .5)
Xn = torch.randint(1, 23, (4, 17))
L_ref = ref(Xn)
agree = []
# (a) tokenwise incremental vs full prefill, at several partial-chunk boundaries
for split in [1, 5, 9, 16]:
    _, st = ref(Xn[:, :split], return_state=True)
    lg = None
    for j in range(split, 17):
        lg, st = ref.step(Xn[:, j], st)
    d = (lg[:, 0] - L_ref[:, -1]).abs()
    agree.append(dict(mode=f'tokenwise from t={split}', max=d.max().item(),
                      rms=d.pow(2).mean().sqrt().item()))
# (b) chunkwise B>1 vs B=1 -- these are DIFFERENT models, so the diff must be large
for B in [2, 4, 8]:
    m2 = RLT(RLTConfig(chunk=B, **cfg)).double().eval(); m2.load_state_dict(ref.state_dict())
    d = (m2(Xn) - L_ref).abs()
    agree.append(dict(mode=f'chunkwise B={B} vs B=1', max=d.max().item(),
                      rms=d.pow(2).mean().sqrt().item()))
    # partial final chunk: 17 is not a multiple of any of these, so this exercises App. I.3
    d2 = (m2(Xn[:, :16]) - m2(Xn)[:, :16]).abs()
    agree.append(dict(mode=f'B={B} partial-chunk prefix consistency', max=d2.max().item(),
                      rms=d2.pow(2).mean().sqrt().item()))
for r in agree:
    print(f"   {r['mode']:<38} max {r['max']:.3e}   rms {r['rms']:.3e}")
OUT['numerical_agreement'] = agree

# ---------------------------------------------------------------------------
print('\n== 6  full-BPTT parameter-gradient agreement (float64, CPU) ==')
def grads(model, X):
    model.zero_grad()
    logits = model(X)
    masked_loss(logits, torch.randint(0, 23, X.shape, generator=torch.Generator().manual_seed(1)),
                torch.ones_like(X, dtype=torch.bool)).backward()
    return torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])

g_ref = grads(ref, Xn)
grow = []
for B in [1, 2, 4, 8]:
    m2 = RLT(RLTConfig(chunk=B, **cfg)).double(); m2.load_state_dict(ref.state_dict())
    g = grads(m2, Xn)
    d = (g - g_ref).abs()
    cos = torch.nn.functional.cosine_similarity(g[None], g_ref[None]).item()
    grow.append(dict(B=B, max=d.max().item(), rms=d.pow(2).mean().sqrt().item(), cosine=cos))
    print(f'   B={B:<3} max|dg| {d.max().item():.3e}   rms {d.pow(2).mean().sqrt().item():.3e}'
          f'   cos(g, g_B1) {cos:.6f}')
OUT['gradient_agreement'] = grow

# ---------------------------------------------------------------------------
print('\n== 7  App. B.3 recurrent path length, counted ==')
depth = []
for split in [(4, 4), (5, 3), (6, 2), (7, 1), (8, 0)]:
    LE, LD = split
    for t in [1, 8, 32, 128]:
        depth.append(dict(split=f'{LE}+{LD}', tokens=t, path_blocks=t * LD,
                          blocks_per_token=LE + LD))
    print(f'   {LE}+{LD}: per token {LE+LD} blocks; path after 128 tokens = {128*LD} decoder blocks')
OUT['recurrent_depth'] = depth

json.dump(OUT, open(A.out, 'w'), indent=1)
print('\nwrote', A.out)
