"""Length generalization from the best ID-validation checkpoint of every run.  Sec 3.5 / App. J.5.

Every (architecture, seed) sees the IDENTICAL test examples at a given task and length -- the test
sets are generated once from the App. J.4 test seed and cached, so the comparison across models is
paired, not merely matched in distribution.
"""
import argparse, glob, json, os, sys, time, traceback
sys.path.insert(0, '/content/RLT_Experiments')

p = argparse.ArgumentParser()
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--tasks', default='parity')
p.add_argument('--src', default='main')
p.add_argument('--out', default='lengthgen')
p.add_argument('--n', type=int, default=256)
a = p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)

import torch
from rlt.eval_gen import load_ckpt, score_length
from rlt import tasks as T

SRC = f'/content/RLT_Experiments/results/{a.src}'
OUT = f'/content/RLT_Experiments/results/{a.out}'
os.makedirs(OUT, exist_ok=True)

for task in a.tasks.split(','):
    for ck_path in sorted(glob.glob(f'{SRC}/{task}__*.pt')):
        base = os.path.basename(ck_path)[:-3]
        out = f'{OUT}/{base}.json'
        if os.path.exists(out):
            print(f'[gpu{a.gpu}] skip {base}', flush=True); continue
        t0 = time.time()
        try:
            model, ck = load_ckpt(ck_path)
            rows = [score_length(model, task, L, n=a.n) for L in T.TEST_LENGTHS[task]]
            _, arch, seed = base.split('__')
            json.dump(dict(task=task, arch=arch, seed=int(seed), ckpt_step=ck['step'],
                           trained_upto=T.TRAINED_UPTO[task], n_per_length=a.n, rows=rows),
                      open(out, 'w'), indent=1)
            print(f"[gpu{a.gpu}] {base:<38} ckpt@{ck['step']:<5} "
                  + ' '.join(f"{r['length']}:{r['final']:.1f}" for r in rows)
                  + f"  {time.time()-t0:.0f}s", flush=True)
            del model; torch.cuda.empty_cache()
        except Exception:
            print(f'[gpu{a.gpu}] FAILED {base}', flush=True); traceback.print_exc()
print(f'[gpu{a.gpu}] DONE', flush=True)
