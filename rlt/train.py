"""Training + evaluation.  Sec. 2.7, Sec. 3.1, App. D.1, App. F.4, App. J.3."""
import json, math, os, time, copy
import torch
import torch.nn.functional as F

from .build import build
from .layers import mup_param_groups
from . import data as D
from . import tasks as T


# ---------------------------------------------------------------------------
# Loss.  App. F.4 step 4: "Normalize each example by its number of selected targets, then average
# examples ...  Token averaging would give more weight to examples with more selected targets."
# ---------------------------------------------------------------------------
def masked_loss(logits, y, m):
    ll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction='none')
    ll = ll.view_as(y) * m                                     # (B, L)
    per_ex = ll.sum(1) / m.sum(1).clamp(min=1)
    return per_ex[m.any(1)].mean()


# ---------------------------------------------------------------------------
# Metrics (App. J.3).
#   token / prefix-token : pooled over every supervised position
#   final                : the LAST supervised position of each example
#   sequence             : every supervised position of an example correct
# ---------------------------------------------------------------------------
@torch.no_grad()
def metrics(logits, y, m):
    pred = logits.argmax(-1)
    corr = (pred == y) & m
    n_tok = m.sum().item()
    tok = corr.sum().item() / max(n_tok, 1)
    last = m.size(1) - 1 - m.flip(1).float().argmax(1)          # index of last True per row
    rows = torch.arange(m.size(0), device=m.device)
    valid = m.any(1)
    fin = (corr[rows, last] & valid).sum().item() / max(valid.sum().item(), 1)
    seq = ((corr.sum(1) == m.sum(1)) & valid).sum().item() / max(valid.sum().item(), 1)
    ll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction='none')
    ll = (ll.view_as(y) * m)
    per_ex = ll.sum(1) / m.sum(1).clamp(min=1)
    return dict(token=100 * tok, final=100 * fin, sequence=100 * seq,
                loss=per_ex[valid].mean().item(), n_tok=n_tok, n_ex=int(valid.sum().item()))


@torch.no_grad()
def evaluate(model, sets, micro=256):
    """Pool counts across the configured validation lengths, then score (App. J.3)."""
    model.eval()
    agg = dict(c_tok=0, n_tok=0, c_fin=0, c_seq=0, n_ex=0, loss=0.0)
    for (L, x, y, m) in sets:
        for i in range(0, x.size(0), micro):
            xb, yb, mb = x[i:i+micro], y[i:i+micro], m[i:i+micro]
            r = metrics(model(xb), yb, mb)
            agg['c_tok'] += r['token'] / 100 * r['n_tok']; agg['n_tok'] += r['n_tok']
            agg['c_fin'] += r['final'] / 100 * r['n_ex']
            agg['c_seq'] += r['sequence'] / 100 * r['n_ex']
            agg['loss'] += r['loss'] * r['n_ex']; agg['n_ex'] += r['n_ex']
    model.train()
    n = max(agg['n_ex'], 1)
    return dict(token=100*agg['c_tok']/max(agg['n_tok'],1), final=100*agg['c_fin']/n,
                sequence=100*agg['c_seq']/n, loss=agg['loss']/n)


# ---------------------------------------------------------------------------
# LR schedule.  Sec 3.1: "The base learning rate rises to 1e-4 over 200 steps, then follows cosine
# decay to 5e-6 at step 2,000."  Warmup is kept at the same 10% fraction of the budget.
# ---------------------------------------------------------------------------
def lr_at(step, total, peak, floor=5e-6, warm_frac=0.10):
    w = max(1, int(warm_frac * total))
    if step < w:
        return peak * (step + 1) / w
    p = (step - w) / max(1, total - w)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * p))


def run(arch, task, seed, *, steps=600, batch=512, peak_lr=1e-4, d_model=128,
        eval_every=25, device='cuda', out=None, val_n=256, micro=None,
        amp=False, save_best=True, quiet=False, extra_cfg=None):
    """One (architecture, task, initialization seed) run.  Returns the record dict."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    model = build(arch, T.VOCAB, d_model=d_model, **(extra_cfg or {})).to(dev)
    cfg = model.cfg
    opt = torch.optim.AdamW(mup_param_groups(model, cfg, peak_lr, 0.1),
                            betas=(0.9, 0.95), eps=1e-8)       # Sec 3.1: (b1,b2)=(0.9,0.95), wd 0.1
    scaler = torch.amp.GradScaler('cuda', enabled=amp)         # T4 is sm_75 -> fp16, never bf16

    xs, ys, ms = D.train_stream(task, steps * batch, dev)
    vsets = D.val_sets(task, val_n, dev)
    micro = micro or batch

    hist, best = [], dict(loss=float('inf'), step=-1, state=None)
    t0 = time.time()
    for step in range(steps):
        for g, lr_mult in zip(opt.param_groups, [cfg.mup_base_width / cfg.d_model, 1.0]):
            g['lr'] = lr_at(step, steps, peak_lr) * lr_mult
        opt.zero_grad(set_to_none=True)
        a = step * batch
        tot = 0.0
        for j in range(0, batch, micro):
            xb, yb, mb = xs[a+j:a+j+micro], ys[a+j:a+j+micro], ms[a+j:a+j+micro]
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=amp):
                loss = masked_loss(model(xb), yb, mb) * (micro / batch)
            scaler.scale(loss).backward()
            tot += loss.item()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # "gradient clipping at norm 1"
        scaler.step(opt); scaler.update()

        if (step + 1) % eval_every == 0 or step == 0:
            ev = evaluate(model, vsets)
            hist.append(dict(step=step + 1, train_loss=tot, grad_norm=float(gn), **ev))
            # App. J.5: select the checkpoint with lowest ID-validation example loss, earliest
            # step on ties.  Test results never participate in selection.
            if save_best and ev['loss'] < best['loss'] - 1e-12:
                best = dict(loss=ev['loss'], step=step + 1,
                            state={k: v.detach().clone() for k, v in model.state_dict().items()})
            if not quiet:
                print(f"  {arch:<10} {task:<14} s{seed} step {step+1:>4}/{steps} "
                      f"loss {tot:.4f} val_final {ev['final']:6.2f}% val_tok {ev['token']:6.2f}%")

    rec = dict(arch=arch, task=task, seed=seed, steps=steps, batch=batch, peak_lr=peak_lr,
               d_model=d_model, n_params=model.n_params(), history=hist,
               best_step=best['step'], best_val_loss=best['loss'],
               wall_s=round(time.time() - t0, 1),
               final=hist[-1] if hist else None)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(rec, open(out, 'w'))
        if save_best and best['state'] is not None:
            torch.save({'state': best['state'], 'arch': arch, 'd_model': d_model,
                        'step': best['step'], 'cfg': cfg.to_dict()},
                       out.replace('.json', '.pt'))
    return rec, model
