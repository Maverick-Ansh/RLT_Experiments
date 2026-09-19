# RLT_Experiments

A from-scratch reproduction of **"Recurrent Looped Transformer" (RLT-1)** by Yifan Zhang, Jichen
Feng and Shihan Qin.

Every module here — the causal encoder, the recurrent decoder, the gated merge, the sliding-window
attention, the encoder-derived memory, the width-muP recipe, all six task generators, the training
loop and every metric — was written from the paper text alone. The paper ships no code.

The full write-up, including what failed, is **[REPORT.md](REPORT.md)**.

## What this found

**The architecture is right.** Reconstructed from prose, it reproduces the parameter counts of
Table 4 for all seven configurations simultaneously (RLT-1 26.10–28.72M, Transformer 25.31M). The
paper never states the FFN type, whether the readout is tied, or how the cross-attention
projections are structured, so those counts are the only independent oracle available — and one
reading satisfies all of them at once. Six formal gates then check the propositions directly in
float64: serving-split invariance (Prop. B.1) and RLT-2's B=1 reduction (App. I.1) both hold to
**0 ulp**, not approximately.

**The central architectural claim reproduces.** RLT-1 learns parity far better than a matched
decoder-only Transformer.

**The paper's headline number does not — and the reason is its evaluation protocol.** The paper
validates parity by pooling only lengths 32, 36 and 40. At that budget both models sit at chance
there and report ~50%, indistinguishable. Scoring the *same checkpoints* at every length in the
3–40 training range shows RLT-1 6+2 solving parity perfectly out to length 12 and decaying to
chance near 28, while the Transformer barely learns it at any length. The paper's own metric
compresses a large real architectural difference into two numbers that differ by less than their
seed spread.

**The mod-5 floor in the paper is wrong.** Bracketed mod-5 is not label-balanced — multiplying by
zero collapses a whole subtree — so a constant predictor scores 26.46%, not the 20% uniform
reference the paper compares against. Parity's true floor is 53.61%, not 50%.

**The cost benchmark the paper specifies but never runs** (App. I.4) is run here: matched
throughput, prefill latency, per-token generation, peak memory, numerical agreement across
execution modes, and the gradient fidelity that chunking trades away.

## Layout

```
rlt/          config, layers, encoder, decoder, model, build, tasks, data, train, eval_gen, viz
scripts/      gates.py            six formal gates (float64, CPU, no training)
              eval_brackets.py    task floors: uniform / majority-class / untrained
              rl_replay.py        App. D.3/D.4 replay and support-condition checks
              cost_bench.py       App. I.4 matched benchmark
              run_sweep.py        resumable sweep runner, one process per GPU
              fig3_parity.py      Fig. 3 reproduction
              fig_parity_length.py  parity resolved by length
              make_report.py      regenerates REPORT.md from results/*.json only
results/      every number in REPORT.md is read back out of these files
figures/
```

## Reproducing

```bash
python scripts/gates.py
python scripts/eval_brackets.py
python scripts/rl_replay.py
python scripts/cost_bench.py          # give it a GPU to itself; it ignores CUDA_VISIBLE_DEVICES
python scripts/run_sweep.py --gpu 0 --tasks parity --archs rlt1_6+2,gpt_8 \
    --seeds 42,43,44 --steps 500 --lr 1e-4 --d_model 512 --tag fig3
python scripts/fig3_parity.py && python scripts/fig_parity_length.py
python scripts/make_report.py
```

Hardware: 2x Tesla T4, FP32. The RLT-1 decoder is a genuine sequential scan over token positions,
so it is launch-bound rather than FLOP-bound — one training step costs ~2.9 s at the benchmark
shape, about 5.7x a matched Transformer. The paper's full grid is roughly 180 GPU-hours.

## Citation

Zhang, Y., Feng, J., Qin, S. *Recurrent Looped Transformer.*
