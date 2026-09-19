"""App. J.7 -- the sixteen-layer addition series, seed 42.

The paper's finding to test: "every RLT-1 split produces zero exact answers out of 256 pairs at
every tested width from nine to 32 digits. The sixteen-layer Transformer produces three exact
answers at nine digits (1.17%) and none at any longer tested width. Teacher-forced answer-token
accuracy is higher."

So: teacher forcing flatters these models badly, and the strict metric (correct answer THEN EOS,
generated greedily) collapses to zero.  Both are measured here, with pointwise 95% Wilson
intervals across test pairs as App. J.7 specifies.
"""
import argparse, json, os, sys, time, traceback
sys.path.insert(0, '/content/RLT_Experiments')

ap = argparse.ArgumentParser()
ap.add_argument('--gpu', type=int, default=0)
ap.add_argument('--archs', default='rlt1_8+8,rlt1_10+6,rlt1_12+4,rlt1_14+2,rlt1_16+0,gpt_16')
ap.add_argument('--steps', type=int, default=500)
ap.add_argument('--batch', type=int, default=512)
ap.add_argument('--lr', type=float, default=4e-4)
ap.add_argument('--d_model', type=int, default=128)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--n_test', type=int, default=256)
A = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(A.gpu)

import torch
from rlt.train import run
from rlt.eval_gen import load_ckpt, score_length, exact_answer_acc, wilson
from rlt import tasks as T

OUT = '/content/RLT_Experiments/results/depth16'
os.makedirs(OUT, exist_ok=True)
GRID = T.TEST_LENGTHS['addition']          # 9, 10, 11, 12, 14, 16, 32

for arch in A.archs.split(','):
    out = f'{OUT}/addition__{arch}__{A.seed}.json'
    if os.path.exists(out):
        print(f'[d16] skip {arch}', flush=True); continue
    t0 = time.time()
    try:
        rec, _ = run(arch, 'addition', A.seed, steps=A.steps, batch=A.batch, peak_lr=A.lr,
                     d_model=A.d_model, eval_every=50, out=out, quiet=True)
        model, ck = load_ckpt(out.replace('.json', '.pt'))
        rows = []
        for L in GRID:
            s = score_length(model, 'addition', L, n=A.n_test)
            ex = exact_answer_acc(model, L, n=A.n_test)
            lo, hi = wilson(round(ex / 100 * A.n_test), A.n_test)
            s.update(exact=ex, exact_lo=lo, exact_hi=hi,
                     exact_k=round(ex / 100 * A.n_test), exact_n=A.n_test)
            rows.append(s)
        rec['gen_rows'] = rows
        rec['n_params'] = model.n_params()
        json.dump(rec, open(out, 'w'))
        print(f"[d16] {arch:<12} params {model.n_params()/1e6:.2f}M  ckpt@{ck['step']:<5} "
              f"ID {rec['final']['token']:.2f}%  | teacher-forced "
              + ' '.join(f"{r['length']}d:{r['token']:.1f}" for r in rows), flush=True)
        print(f"{'':<19}exact-answer " + ' '.join(f"{r['length']}d:{r['exact']:.2f}%"
                                                  for r in rows), flush=True)
        del model; torch.cuda.empty_cache()
    except Exception:
        print(f'[d16] FAILED {arch}', flush=True); traceback.print_exc(); sys.stdout.flush()
print('[d16] DONE', flush=True)
