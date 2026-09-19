"""Assert the paper's task RULES, not tensor shapes.

A generator bug quietly becomes "a hard dataset" and confounds every downstream comparison, so
each task is verified against an INDEPENDENT reference implementation -- S5 products are re-derived
by composing tuples directly rather than through the Cayley table the generator uses, and both
mod-5 variants are re-evaluated with python's own parser.
"""
import sys, itertools, collections
import numpy as np
sys.path.insert(0, '/content/RLT_Experiments')
from rlt import tasks as T

fails = []
def check(name, cond, extra=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond: fails.append(name)

rng = np.random.default_rng(0)
print('S5 group')
check('120 distinct permutations', len(set(T.PERMS)) == 120)
check('Cayley table is a Latin square', all(len(set(T.CAYLEY[i])) == 120 for i in range(120)))
check('identity is the unit', all(T.CAYLEY[T.IDENTITY, j] == j and T.CAYLEY[j, T.IDENTITY] == j
                                  for j in range(120)))
swaps = T.TRANSPOSITIONS[1:]
check('swaps variant = id + 10 transpositions', len(T.TRANSPOSITIONS) == 11 and len(set(swaps)) == 10)
check('each swap moves exactly 2 letters',
      all(sum(1 for i, v in enumerate(T.PERMS[s]) if v != i) == 2 for s in swaps))
check('each swap is an involution', all(T.CAYLEY[s, s] == T.IDENTITY for s in swaps))
check('associativity (500 random triples)',
      all(T.CAYLEY[T.CAYLEY[a, b], c] == T.CAYLEY[a, T.CAYLEY[b, c]]
          for a, b, c in rng.integers(0, 120, size=(500, 3))))

print('\naddition')
x, y, m = T.make('addition', 400, T.ADD_SEED)
def dec_int(toks):
    return int(''.join(str(t - T.DIGIT0) for t in toks)[::-1])
bad = 0
for r in range(400):
    row = [t for t in x[r] if t != T.PAD]
    p, e = row.index(T.PLUS), row.index(T.EQ)
    a, b, s = dec_int(row[1:p]), dec_int(row[p+1:e]), dec_int(row[e+1:-1])
    if a + b != s: bad += 1
check('a + b == sum for all 400', bad == 0, f'({bad} bad)')
check('mask covers answer tokens AND EOS', all(y[r][m[r]][-1] == T.EOS for r in range(400)))
check('mask excludes prompt and padding',
      all(not m[r][:list(x[r]).index(T.EQ)].any() for r in range(400)))
check('no target is PAD inside the mask', (y[m] != T.PAD).all())
check('operand widths span 1..8',
      set(len(str(dec_int([t for t in x[r] if t != T.PAD][1:list(x[r]).index(T.PLUS)])))
          for r in range(400)) >= {1, 8})
# 32-digit operands overflow numpy int64; the generator must stay in python big ints.
x32, y32, m32 = T.make('addition', 64, T.TEST_SEED_FORMAL, length=32)
bad32 = 0
for r in range(64):
    row = [t for t in x32[r] if t != T.PAD]
    p, e = row.index(T.PLUS), row.index(T.EQ)
    a, b, s = dec_int(row[1:p]), dec_int(row[p+1:e]), dec_int(row[e+1:-1])
    if a + b != s or len(str(a)) != 32 or len(str(b)) != 32: bad32 += 1
check('32-digit operands: exact width and exact sum', bad32 == 0, f'({bad32} bad)')

print('\nparity')
x, y, m = T.make('parity', 2000, T.TRAIN_SEED_FORMAL)
ok = True
for r in range(2000):
    eq = list(x[r]).index(T.EQ)
    bits = x[r][1:eq] - T.DIGIT0
    ok &= (y[r][eq] - T.DIGIT0) == bits.sum() % 2
    ok &= m[r].sum() == 1 and m[r][eq]
check('target == XOR of bits, exactly one supervised position', ok)
lab = (y[m] - T.DIGIT0)
check('label balance ~50%', abs(lab.mean() - .5) < .05, f'(p1={lab.mean():.3f})')
lens = [list(x[r]).index(T.EQ) - 1 for r in range(2000)]
check('lengths span 3..40', min(lens) == 3 and max(lens) == 40)
check('fixed length request honoured',
      all(list(r).index(T.EQ) - 1 == 40 for r in T.make('parity', 50, 1, length=40)[0]))

print('\nmod5 flat (independent re-evaluation with a precedence parser)')
x, y, m = T.make('mod5_flat', 1500, T.TRAIN_SEED_FORMAL)
def ref_flat(toks):
    """Independent reference: textual eval with python's own precedence, then mod 5."""
    s = ''.join({T.PLUS: '+', T.MINUS: '-', T.TIMES: '*'}.get(t, str(t - T.DIGIT0)) for t in toks)
    return eval(s) % 5
ok, dist = True, collections.Counter()
for r in range(1500):
    eq = list(x[r]).index(T.EQ)
    v = ref_flat(x[r][1:eq])
    ok &= (y[r][eq] - T.DIGIT0) == v
    dist[v] += 1
check('multiplication precedence matches a real parser', ok)
check('label ~uniform over 5', max(dist.values()) / 1500 < .30, f'{dict(dist)}')
check('expression lengths odd and in 3..39',
      all((list(x[r]).index(T.EQ) - 1) % 2 == 1 and 3 <= list(x[r]).index(T.EQ) - 1 <= 39
          for r in range(1500)))

print('\nmod5 brackets (independent re-evaluation)')
x, y, m = T.make('mod5_brackets', 1500, T.TRAIN_SEED_FORMAL)
def ref_br(toks):
    s = ''.join({T.PLUS: '+', T.MINUS: '-', T.TIMES: '*', T.LPAR: '(', T.RPAR: ')'}
                .get(t, str(t - T.DIGIT0)) for t in toks)
    return eval(s) % 5
ok, bal, dist = True, True, collections.Counter()
for r in range(1500):
    eq = list(x[r]).index(T.EQ)
    toks = x[r][1:eq]
    d = 0
    for t in toks:
        d += int(t == T.LPAR) - int(t == T.RPAR)
        bal &= d >= 0
    bal &= d == 0
    v = ref_br(toks)
    ok &= (y[r][eq] - T.DIGIT0) == v
    dist[v] += 1
check('parentheses balanced', bal)
check('value matches a real parser', ok)
check('label ~uniform over 5', max(dist.values()) / 1500 < .32, f'{dict(dist)}')
check('lengths on the 4k-5 lattice, 3..39',
      all((list(x[r]).index(T.EQ) - 1) % 4 == 3 for r in range(1500)))
for L in T.TEST_LENGTHS['mod5_brackets']:
    xx = T.make('mod5_brackets', 16, T.TEST_SEED_FORMAL, length=L)[0]
    check(f'test grid length {L} hit exactly',
          all(list(r).index(T.EQ) - 1 == L for r in xx))

print('\nS5 (running product re-derived by composing tuples directly)')
for variant in ['s5_standard', 's5_swaps']:
    x, y, m = T.make(variant, 300, T.TRAIN_SEED_FORMAL)
    ok = True
    for r in range(300):
        st = (0, 1, 2, 3, 4)
        for t in range(32):
            p = T.PERMS[int(x[r, t + 1]) - T.PERM0]
            st = tuple(st[p[i]] for i in range(5))      # st o p, same order as CAYLEY[st, p]
            ok &= T.PERM_ID[st] == int(y[r, t]) - T.PERM0
    check(f'{variant}: running product correct at every prefix', ok)
    check(f'{variant}: all 32 positions supervised', m.sum(1).min() == 32 and m.sum(1).max() == 32)
    n_sym = len(set(x[:, 1:].ravel().tolist()))
    check(f'{variant}: alphabet size', n_sym == (11 if variant == 's5_swaps' else 120), f'({n_sym})')

print('\ntrain/val/test streams are disjoint draws')
a = T.make('parity', 256, T.TRAIN_SEED_FORMAL, length=40)[0]
b = T.make('parity', 256, T.VAL_SEED_FORMAL,   length=40)[0]
c = T.make('parity', 256, T.TEST_SEED_FORMAL,  length=40)[0]
check('train != val != test', not np.array_equal(a, b) and not np.array_equal(b, c))
check('same seed reproduces exactly',
      np.array_equal(b, T.make('parity', 256, T.VAL_SEED_FORMAL, length=40)[0]))

print('\n' + ('ALL TASK CHECKS PASSED' if not fails else f'FAILURES: {fails}'))
sys.exit(1 if fails else 0)
