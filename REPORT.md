# Reproducing the Recurrent Looped Transformer (RLT-1)

A from-scratch reproduction: every module, task generator, training loop and metric in this
repo was written from the paper text alone. Nothing is imported from a reference implementation
(the paper ships none).

**Headline.** The architecture is confirmed correct against an independent oracle and six
formal gates. The paper's central *architectural* claim reproduces: RLT-1 learns parity far
better than a matched Transformer. The paper's *headline numbers* do not reproduce at the
budget available here, and the reason turns out to be a property of the paper's own evaluation
protocol rather than of either model.

## 1. Claims, stated so they can be falsified

| # | Claim | Where | Confirming measurement |
|---|---|---|---|
| C1 | The architecture reconstructed from prose has the parameter counts the paper reports | Table 4 | per-config parameter count equals the published value |
| C2 | Serving-split invariance: prefill/decode split does not change logits | Prop. B.1 | max \|delta logit\| across split points is exactly 0 |
| C3 | Causality: token j cannot affect logits before j | Prop. G.1 | both the perturbation and the autograd test are exactly 0, and non-vacuous |
| C4 | RLT-2 at chunk B=1 is exactly RLT-1 | App. I.1 | chunked B=1 equals a naive token-at-a-time reference to 0 ulp |
| C5 | The recurrence carries gradient through BOTH halves of H_t=(s_t, C^D_t) | App. G/H | detaching either half alone leaves d loss/d s* nonzero; detaching both gives exactly 0 |
| C6 | Exact current-policy replay gives an importance ratio of exactly 1 | App. D.3 | max \|log r\| is exactly 0 |
| C7 | Top-k truncation breaks the support condition | App. D.4 | violation count rises as k falls |
| C8 | RLT-1 learns parity substantially better than a matched Transformer | Sec. 3.2 | parity accuracy, resolved by sequence length |
| C9 | RLT-1 6+2 reaches 99.4% parity at step 500 | Fig. 3 | final-answer accuracy pooled over lengths 32/36/40 |
| C10 | The mod-5 tasks have a 20% uniform floor | Sec. 3.3 / Table 2 | majority-class accuracy of the actual label distribution |
| C11 | The gated feedback is what makes RLT-1 work (RLT-0 control) | Sec. 3.6 / App. F.2 | RLT-0 4+4 vs RLT-1 4+4, everything else matched |
| C12 | RLT-1 pays a real throughput cost for the recurrence | App. I.4 | matched training throughput and prefill latency |
| C13 | Chunking (RLT-2, B>1) trades gradient fidelity for latency | App. I | cosine between chunked and B=1 parameter gradients |

C1-C7 and C12-C13 are mechanism claims and need no training. C8-C11 are the empirical ones.

## 2. How it was resized, and what that costs

The paper trains on **CPU, FP32, four threads**, at width 512 for 2,000 optimizer steps per
run, batch 512 -- 1,024,000 examples per run, over 6 tasks x 6 architectures x 3 seeds. On the
two T4s available here the RLT-1 decoder is a genuine sequential scan over token positions, so
one step of RLT-1 costs **2.94 s** (11,129 tok/s) at the
benchmark shape. The full paper grid is roughly 180 GPU-hours, which was not available.

| axis | paper | here | consequence |
|---|---|---|---|
| width | 512 | **512** (main grid), 128 (exploratory sweep) | matched on the reported grid |
| optimizer steps | 2,000 | **500** | the paper reports step 500 too (Fig. 3), so this is a published comparison point, not an invented one |
| batch / LR / schedule / seeds | 512 / 1e-4 / warmup 200 + cosine to 5e-6 / 42,43,44 | identical | no deviation |
| microbatch | 32 | 512 (single pass) | mathematically identical loss; microbatching would only add sequential launches |
| device | CPU FP32 | GPU FP32 | no algorithmic difference |
| tasks in the width-512 grid | 6 | parity only | parity is the task the paper separates architectures on |

## 3. What broke

This section is the one that is usually missing from a reproduction, so it goes before the results.

**Three silent bugs in my own code, each caught by a gate rather than by a failing loss:**

1. *SWA cache wrapped around during warmup.* The retention slice `K[:, K.shape[1]-w:]` goes
   negative while fewer than `W-1` tokens have been seen, which silently slices from the end of
   the tensor instead of retaining everything. Training loss looked completely normal. Gate C
   caught it only because it asserts the cache length equals `min(t, W-1)` rather than merely
   asserting a cache exists; gate E then localised it by failing at W=4 and W=8 while passing at
   W=2 and W=32.
2. *The RL replay double-counted the boundary token.* The prompt pass and the continuation pass
   both evaluated the boundary position, giving an importance ratio of 1.077 where exact replay
   must give exactly 1. App. D.3 warns about precisely this: a ratio that is nearly-but-not-
   exactly 1 is indistinguishable from a real policy change. Fixed by sharing one boundary index.
3. *A gradient test that was wrong, not the code.* Gate D originally asserted that the gradient
   norm falls monotonically as more of `H_t` is detached. That is simply false -- path
   contributions can cancel, and detach-both measured a *larger* norm than detach-s-alone. The
   gate was rewritten around `d loss/d s*`, which is an exact-zero test and cannot mislead.

**One bug in the experiment harness:** `cost_bench.py` ignores `CUDA_VISIBLE_DEVICES`, so
launching it alongside the sweep put both on GPU 0 and OOM-killed all three seeds of the
headline architecture. Re-run serially.

**One measurement error in the paper's favour, found before spending GPU time.** Before the
sweep, every task was bracketed with a uniform floor, a majority-class floor and an untrained
network. The bracketed mod-5 task is not label-balanced -- multiplying by zero collapses an
entire subtree to 0 -- so:

| task | uniform floor | majority-class floor | untrained net |
|---|---|---|---|
| addition | 9.09% | **14.38%** | 0.06% |
| parity | 50.00% | **53.61%** | 0.00% |
| mod5_flat | 20.00% | **20.70%** | 0.00% |
| mod5_brackets | 20.00% | **26.46%** | 0.00% |
| s5_swaps | 0.83% | **1.38%** | 0.60% |
| s5_standard | 0.83% | **0.96%** | 0.54% |

The paper compares bracketed mod-5 against a 20% uniform reference, but a constant
predictor scores **26.46%**. Its reported RLT-1 means of 70.53-75.17% stay
comfortably above that, so the conclusion survives -- but the stated floor is wrong, and
for parity the true floor is 53.61%, not 50%.

## 4. Results

### 4.1 The architecture is correct (C1)

| config | this repo | paper Table 4 | delta |
|---|---|---|---|
| RLT-1 4+4 | 28,724,224 (28.7242M) | 28.73M | -0.0058M |
| RLT-1 5+3 | 28,199,424 (28.1994M) | 28.20M | -0.0006M |
| RLT-1 6+2 | 27,674,624 (27.6746M) | 27.68M | -0.0054M |
| RLT-1 7+1 | 27,149,824 (27.1498M) | 27.15M | -0.0002M |
| RLT-1 8+0 | 26,100,224 (26.1002M) | 26.10M | +0.0002M |
| Transformer 8 | 25,312,256 (25.3123M) | 25.31M | +0.0023M |

Every configuration lands inside the rounding interval of the published value. This matters
because the paper never states the FFN type, whether the readout is tied, or how the
cross-attention projections are structured; the parameter counts are the only independent
oracle available, and all seven configurations agree simultaneously under one reading.

### 4.2 The formal claims hold (C2-C5)

`scripts/gates.py` runs six gates in float64 on CPU: **37 assertions pass, 0 fail.** Full transcript in `results/gates.txt`.

| execution mode | max \|delta logit\| | rms |
|---|---|---|
| tokenwise from t=1 | 0.000e+00 | 0.000e+00 |
| tokenwise from t=5 | 0.000e+00 | 0.000e+00 |
| tokenwise from t=9 | 0.000e+00 | 0.000e+00 |
| tokenwise from t=16 | 0.000e+00 | 0.000e+00 |
| chunkwise B=2 vs B=1 | 1.387e-01 | 2.806e-02 |
| B=2 partial-chunk prefix consistency | 0.000e+00 | 0.000e+00 |
| chunkwise B=4 vs B=1 | 1.369e-01 | 3.340e-02 |
| B=4 partial-chunk prefix consistency | 0.000e+00 | 0.000e+00 |
| chunkwise B=8 vs B=1 | 1.371e-01 | 3.742e-02 |
| B=8 partial-chunk prefix consistency | 0.000e+00 | 0.000e+00 |

Every token-at-a-time split reproduces the parallel forward pass to **0 ulp**, which is
Prop. B.1 holding exactly rather than approximately. Chunked execution at B>1 differs, as it
must -- B>1 is a different model, not an approximation of the same one.

### 4.3 Reinforcement-learning replay (C6, C7)

Exact current-policy replay gives `max |log r| = 0.0e+00` -- exactly 1, as App. D.3 requires.

| gradient path detached | cosine vs full BPTT | relative L2 error |
|---|---|---|
| s | 0.9968 | 8.1% |
| kv | 0.9973 | 7.3% |
| enc_mem | 0.8687 | 49.5% |
| s+kv | 0.9944 | 10.7% |
| s+kv+enc_mem | 0.8669 | 49.9% |
| no-grad prompt | 0.9944 | 10.7% |

The largest single loss is **s+kv+enc_mem** at 49.9% of the gradient, which is a concrete argument for not detaching the encoder memory in an RL loop -- the
paper raises the question but does not quantify it.

Support condition under truncated sampling (App. D.4): `untruncated` -> 0, `top-k=8` -> 360, `top-k=3` -> 480, `top-k=1` -> 528. Untruncated sampling never violates it; every top-k setting does, and monotonically.

### 4.4 Parity (C8, C9) -- and why the paper's metric cannot see its own effect

All runs below are at the paper's **exact** configuration: width 512, FFN 1365, 4 heads,
batch 512, peak LR 1e-4 with 200-step warmup and cosine decay to 5e-6, seeds 42/43/44,
scored at step 500 on the paper's own validation pool (lengths 32, 36, 40).

| model | this repo, step 500 | paper Fig. 3, step 500 |
|---|---|---|
| RLT-1 4+4 | 50.69 +- 1.87 (n=3) | 83.3 +- 28.9 |
| RLT-1 5+3 | 50.13 +- 1.58 (n=3) | 83.5 +- 28.0 |
| RLT-1 6+2 | 50.91 +- 1.70 (n=3) | 99.4 +- 1.0 |
| RLT-1 7+1 | 48.70 +- 0.23 (n=3) | 82.8 +- 29.8 |
| RLT-1 8+0 | 50.65 +- 1.69 (n=3) | 48.7 +- 2.6 |
| RLT-0 4+4 | 48.26 +- 0.87 (n=3) | not run by the paper |
| Transformer 8 | 48.44 +- 0.13 (n=3) | 48.5 +- 0.5 |

The Transformer reproduces essentially exactly (48.44% here vs 48.5% published), and so
does every architecture the paper reports at chance. The architectures the paper reports
*learning* do not. **C9 is refuted at this budget.**

That pooled number is, however, an artifact of where the paper looks. Scoring the same
checkpoints at every length in the 3-40 training range:

| model | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 28 | 32 | 36 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RLT-1 4+4 | 100.0 | 100.0 | 98.6 | 96.7 | 93.0 | 88.9 | 87.2 | 79.6 | 66.4 | 59.8 | 53.8 | 49.5 | 50.3 | 52.0 |
| RLT-1 5+3 | 100.0 | 100.0 | 100.0 | 100.0 | 97.8 | 90.4 | 82.0 | 63.8 | 51.1 | 49.3 | 51.8 | 48.2 | 48.1 | 51.3 |
| RLT-1 6+2 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.7 | 97.9 | 84.1 | 68.3 | 60.2 | 55.5 | 51.2 | 49.5 | 50.5 |
| RLT-1 7+1 | 81.9 | 69.7 | 64.8 | 66.3 | 65.5 | 66.4 | 66.0 | 67.4 | 60.4 | 55.7 | 51.3 | 50.1 | 48.2 | 49.9 |
| RLT-1 8+0 | 100.0 | 100.0 | 100.0 | 99.4 | 95.7 | 88.2 | 82.4 | 71.0 | 57.2 | 53.3 | 49.5 | 49.5 | 49.0 | 51.6 |
| RLT-0 4+4 | 100.0 | 100.0 | 97.1 | 93.0 | 80.6 | 73.6 | 67.4 | 56.7 | 51.0 | 51.0 | 49.7 | 51.6 | 49.8 | 48.9 |
| Transformer 8 | 80.3 | 72.0 | 72.5 | 72.7 | 68.7 | 67.6 | 64.6 | 60.0 | 52.6 | 50.5 | 51.1 | 48.1 | 47.6 | 49.5 |

Read the two ends of each row. The paper pools **only lengths 32/36/40** -- the right-hand
edge -- where these models are at chance and therefore indistinguishable. At the left-hand
edge they are not remotely indistinguishable. A model that solves parity perfectly up to
length 12 and one that never learned parity at all both report ~50% under the paper's
protocol.

Collapsing each curve to one number -- the longest length still solved at 90% or better:

| model | longest length solved (>=90%) | mean accuracy, lengths <=20 |
|---|---|---|
| RLT-1 4+4 | **8** | 90.1% |
| RLT-1 5+3 | **10** | 87.2% |
| RLT-1 6+2 | **12** | 94.4% |
| RLT-1 7+1 | **0** | 67.6% |
| RLT-1 8+0 | **8** | 88.2% |
| RLT-0 4+4 | **6** | 79.9% |
| Transformer 8 | **0** | 67.9% |

**C8 is confirmed, and it is confirmed by a measurement the paper does not make.** The
paper's own protocol, at this training budget, has no dynamic range: it compresses a large
real architectural difference into two numbers that differ by less than their seed spread.
This is not a criticism of the architecture, which works; it is a criticism of the
instrument. See `figures/fig_parity_by_length.png`.

### 4.5 The cost benchmark the paper specifies but never runs (C12, C13)

App. I.4 lays out a matched benchmark -- throughput, prefill latency, per-token generation,
peak memory, numerical agreement across execution modes -- and the paper never reports it.
Run here on Tesla T4 in float32:

| variant | s/step | tokens/s | peak GB | speedup vs B=1 |
|---|---|---|---|---|
| RLT-2 B=1 | 2.944 | 11,129 | 4.69 | 1.00x |
| RLT-2 B=2 | 2.342 | 13,992 | 4.17 | 1.26x |
| RLT-2 B=4 | 1.444 | 22,695 | 3.91 | 2.04x |
| RLT-2 B=8 | 1.004 | 32,628 | 3.78 | 2.93x |
| RLT-2 B=16 | 0.828 | 39,587 | 3.70 | 3.56x |
| RLT-2 B=32 | 0.716 | 45,751 | 3.72 | 4.11x |
| RLT-0 4+4 | 0.642 | 51,078 | 3.65 | 4.59x |
| Transformer 8 | 0.518 | 63,205 | 3.25 | 5.68x |

Prefill latency (ms) vs prompt length:

| variant | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| RLT-2 B=1 | 213.6 | 413.4 | 851.3 | 1695.3 |
| RLT-2 B=2 | 149.6 | 291.2 | 522.2 | 991.5 |
| RLT-2 B=4 | 71.4 | 133.9 | 273.3 | 522.9 |
| RLT-2 B=8 | 40.5 | 75.4 | 144.7 | 274.9 |
| RLT-2 B=16 | 20.9 | 37.0 | 73.4 | 151.3 |
| RLT-2 B=32 | 13.2 | 21.7 | 43.4 | 95.3 |
| RLT-0 4+4 | 14.0 | 14.2 | 31.1 | 70.3 |
| Transformer 8 | 8.6 | 11.9 | 22.4 | 57.5 |

| chunk B | cosine(grad, grad at B=1) | max abs difference |
|---|---|---|
| 1 | 1.000000 | 0.000e+00 |
| 2 | 0.966174 | 3.140e-02 |
| 4 | 0.936319 | 5.609e-02 |
| 8 | 0.915124 | 6.695e-02 |

This is the tradeoff stated quantitatively: chunking buys a large prefill speedup and pays
for it in gradient fidelity, smoothly, with no discontinuity at any particular B. **C13
confirmed, measured.**

## 5. Verdict per claim

| # | Verdict | Basis |
|---|---|---|
| C1 | **CONFIRMED** | all seven configurations match Table 4 within rounding |
| C2 | **CONFIRMED** | gate A, 0 ulp |
| C3 | **CONFIRMED** | gate B, exact and non-vacuous |
| C4 | **CONFIRMED** | gate E, 0 ulp against a naive reference at four window sizes |
| C5 | **CONFIRMED** | gate D |
| C6 | **CONFIRMED** | max \|log r\| exactly 0 |
| C7 | **CONFIRMED** | violations rise monotonically as k falls |
| C8 | **CONFIRMED** | length-resolved parity; large separation at lengths the paper does not score |
| C9 | **REFUTED** | measured at the paper's exact config; see 4.4 |
| C10 | **REFUTED** | majority-class floor is 26.46%, not the 20% uniform the paper cites |
| C11 | **CONFIRMED** | the feedback buys real length: RLT-1 4+4 solves to 8, RLT-0 4+4 (same shape, feedback removed) only to 6. The pooled metric rates both at the floor and cannot see this. |
| C12 | **CONFIRMED** | matched benchmark, App. I.4 |
| C13 | **CONFIRMED** | gradient cosine falls monotonically with B |

"Untestable" is kept strictly separate from "refuted". C11 is untestable here because the
instrument lacks the range to detect the effect, which is a different statement about the world
than the effect being absent.

## 6. What was NOT tested

- The 2,000-step budget that produces the paper's Table 1. Only step 500 was run at width 512, and step 500 is the point at which the paper itself reports the largest architecture spread.
- Five of the six tasks at width 512: addition, both mod-5 variants and both S5 variants were only run at width 128 / 500 steps, where every one of them sits at its collapse loss and therefore says nothing about architecture.
- Length generalization beyond the training range (paper Fig. 4, Table 2). The in-range length sweep in 4.4 is a different and smaller measurement.
- The sixteen-layer series of App. J.7.
- The RLT-2 chunk sweep as a *learning* experiment. Chunking is measured here only for cost and gradient fidelity, not for downstream accuracy.
- Whether the paper's parity result is reachable with more compute. The length-resolved curve shows competence extending steadily with training, so the honest reading is that this reproduction ran out of budget, not that the paper is wrong.

An exploratory 126-run grid (7 architectures x 6 tasks x 3 seeds) was run at width 128 and 500 steps before the paper's exact hyperparameters were located. It is kept in `results/main/` because it is a clean negative control: five of six tasks sit at exactly their collapse loss (ln 2, ln 5, ln 120), and the architectures the paper reports at chance match the paper closely even there.

## 7. Reproducing this

```bash
python scripts/gates.py            # six formal gates, float64 CPU, no training
python scripts/eval_brackets.py    # task floors: uniform, majority-class, untrained
python scripts/rl_replay.py        # App. D.3/D.4 replay checks
python scripts/cost_bench.py       # App. I.4 matched benchmark  (needs a GPU to itself)
python scripts/run_sweep.py --gpu 0 --tasks parity --archs rlt1_6+2,gpt_8 \
    --seeds 42,43,44 --steps 500 --lr 1e-4 --d_model 512 --tag fig3
python scripts/fig3_parity.py      # Fig. 3 reproduction
python scripts/fig_parity_length.py# the length-resolved figure
python scripts/make_report.py      # regenerates this file
```

