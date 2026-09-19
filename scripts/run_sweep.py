"""Detached sweep runner: one process per GPU, resumable, logs to a file.

Usage:  python scripts/run_sweep.py --gpu 0 --tasks parity,s5_swaps --steps 500 --tag main
Existing result JSONs are skipped, so a killed sweep resumes instead of restarting.
`--chunk B` drives the RLT-2 chunk size of App. I (B=1 is RLT-1).
"""
import argparse, os, sys, json, time, traceback
sys.path.insert(0, '/content/RLT_Experiments')

p = argparse.ArgumentParser()
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--tasks', default='parity')
p.add_argument('--archs', default='rlt1_4+4,rlt1_5+3,rlt1_6+2,rlt1_7+1,rlt1_8+0,gpt_8')
p.add_argument('--seeds', default='42,43,44')
p.add_argument('--steps', type=int, default=500)
p.add_argument('--batch', type=int, default=512)
p.add_argument('--lr', type=float, default=4e-4)
p.add_argument('--d_model', type=int, default=128)
p.add_argument('--eval_every', type=int, default=25)
p.add_argument('--tag', default='main')
p.add_argument('--save_best', type=int, default=1)
p.add_argument('--chunk', type=int, default=None, help='RLT-2 chunk size B (App. I); 1 == RLT-1')
a = p.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
import torch
from rlt.train import run

OUT = f'/content/RLT_Experiments/results/{a.tag}'
os.makedirs(OUT, exist_ok=True)
extra = {'chunk': a.chunk} if a.chunk is not None else None
suffix = f'__B{a.chunk}' if a.chunk is not None else ''

jobs = [(t, ar, s) for t in a.tasks.split(',')
        for ar in a.archs.split(',') for s in [int(v) for v in a.seeds.split(',')]]
print(f'[gpu{a.gpu}] {len(jobs)} jobs, {a.steps} steps x batch {a.batch}, lr {a.lr:g}'
      + (f', chunk B={a.chunk}' if a.chunk else ''), flush=True)

for i, (task, arch, seed) in enumerate(jobs):
    out = f'{OUT}/{task}__{arch}{suffix}__{seed}.json'
    if os.path.exists(out):
        print(f'[gpu{a.gpu}] skip {os.path.basename(out)}', flush=True); continue
    t0 = time.time()
    try:
        rec, _ = run(arch, task, seed, steps=a.steps, batch=a.batch, peak_lr=a.lr,
                     d_model=a.d_model, eval_every=a.eval_every, out=out,
                     save_best=bool(a.save_best), quiet=False, extra_cfg=extra)
        f = rec['final']
        print(f"[gpu{a.gpu}] {i+1}/{len(jobs)} {task:<14}{arch:<10}{suffix:<5}s{seed}  "
              f"final={f['final']:6.2f}%  token={f['token']:6.2f}%  loss={f['loss']:.4f}  "
              f"best@{rec['best_step']}  {time.time()-t0:.0f}s", flush=True)
    except Exception:
        print(f'[gpu{a.gpu}] FAILED {task} {arch} {seed}', flush=True)
        traceback.print_exc(); sys.stdout.flush()
    torch.cuda.empty_cache()
print(f'[gpu{a.gpu}] DONE', flush=True)
