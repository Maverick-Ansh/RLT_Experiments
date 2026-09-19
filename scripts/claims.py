"""The falsifiable claims this reproduction tests, with the number that would settle each one.

Written BEFORE the sweep ran.  Each entry carries the paper's own figure and the criterion, so a
verdict is read off the data rather than argued for afterwards.
"""
CLAIMS = [
 dict(id='C1', kind='mechanism', where='Table 4 / App. J.1',
      claim='The architecture is fully determined by the published parameter counts: SwiGLU FFN, '
            'tied readout, cross-attention with query/output projections only, and no memory '
            'projection in the 8+0 split.',
      criterion='All six counts reproduced to <0.01M, and every difference between splits exact.'),
 dict(id='C2', kind='mechanism', where='Prop. B.1',
      claim='Moving the prompt/response split changes neither the states nor the next-token '
            'distribution.',
      criterion='max |logit difference| < 1e-10 in float64 across four split points.'),
 dict(id='C3', kind='mechanism', where='Prop. G.1',
      claim='H_t depends only on x_{1:t}.',
      criterion='Perturbing token j leaves every logit before j bit-identical (exactly 0.0), and '
                'd(logits_t)/d(e_j) == 0 for j > t.'),
 dict(id='C4', kind='mechanism', where='App. G / H',
      claim='Detaching only s_t leaves gradient paths through decoder KV and vice versa; '
            'detaching both cuts the recurrence.',
      criterion='d loss/d s_star nonzero under each single detach, EXACTLY zero under both.'),
 dict(id='C5', kind='mechanism', where='App. I.1',
      claim='RLT-2 with chunk size B=1 is exactly RLT-1.',
      criterion='max |logit difference| vs a naive token-at-a-time reference < 1e-11, at W in '
                '{2,4,8,32}.'),
 dict(id='C6', kind='headline', where='Sec. 3.2, Fig. 3 (left panel)',
      claim='At 256,000 training examples the RLT-1 depth splits learn parity faster than the '
            'decoder-only Transformer; 6+2 is furthest ahead and 8+0 stays near chance.',
      criterion='Best RLT-1 split mean > Transformer mean by more than the pooled SD, at step 500.'),
 dict(id='C7', kind='headline', where='Sec. 3.5, Table 2',
      claim='Parity length generalization differs across depth splits; 5+3 and 7+1 hold at 256 '
            'bits while the Transformer falls to chance.',
      criterion='At the longest tested length, some RLT-1 split beats the Transformer by more '
                'than the pooled SD AND beats the majority-class floor.'),
 dict(id='C8', kind='headline', where='Sec. 3.5, Table 2',
      claim='On swaps-S5 the RLT-1 splits with larger decoders extrapolate to far longer '
            'sequences than the Transformer (4+4: 55.70% at 512 operations vs 0.85%).',
      criterion='Some RLT-1 split > Transformer + pooled SD at the longest tested length.'),
 dict(id='C9', kind='headline', where='Sec. 3.4',
      claim='Standard S5 stays near the uniform reference for every architecture.',
      criterion='Every architecture within a few points of 0.83% final-state accuracy.'),
 dict(id='C10', kind='headline', where='Sec. 3.5',
      claim='Teacher-forced addition accuracy falls once operand width leaves the trained range, '
            'for every architecture.',
      criterion='Monotone decline from 9 to 32 digits for all architectures.'),
 dict(id='C11', kind='headline', where='Sec. 3.3',
      claim='Flat modular arithmetic has large initialization variability (paper SDs 7-46 points).',
      criterion='Sample SD across three seeds notably larger on mod5_flat than on mod5_brackets.'),
 dict(id='C12', kind='headline', where='App. J.7',
      claim='In the sixteen-layer series, greedy exact-answer accuracy is ~0 beyond the trained '
            'range for every model, while teacher forcing looks far healthier.',
      criterion='exact-answer ~0% at every tested width, teacher-forced token accuracy much higher.'),
 dict(id='C13', kind='mechanism', where='App. D.3 / D.4',
      claim='Exact current-policy replay reproduces the sampler bit-for-bit; no_grad prompt replay '
            'keeps the forward values and changes the gradient; top-k sampling breaks the support '
            'condition.',
      criterion='max|log r| == 0 at unchanged parameters; forward identical but gradient cosine '
                '< 1; >0 support violations under top-k.'),
 dict(id='C14', kind='new', where='App. I.4 (specified, never run)',
      claim='Increasing the RLT-2 chunk size B buys throughput by updating the state less often, '
            'and costs accuracy on a task whose difficulty is carrying per-token state.',
      criterion='Monotone throughput gain in B, with accuracy falling toward RLT-0.'),
 dict(id='C15', kind='new', where='Sec. 3.6 ("matched runs are needed to isolate feedback")',
      claim='RLT-0 4+4 vs RLT-1 4+4 isolates the feedback wire alone, holding the encoder/decoder '
            'split, attention structure and parameter budget fixed.',
      criterion='Sign and size of the RLT-1 minus RLT-0 gap, per task, against the seed spread.'),
]
