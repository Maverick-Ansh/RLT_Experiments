"""The six algorithmic tasks of Sec. 3.1, generated from scratch with numpy.

Every generator returns (x, y, m):
    x : (B, L) int64   input tokens
    y : (B, L) int64   target for the prediction made AFTER reading x[:, i]
    m : (B, L) bool    which positions are supervised
This matches Eq. (2.6): p(x_{t+1} | x_{1:t}) = softmax(W_o RMSNorm_o(s_t)), i.e. the logits at
position i score y[:, i].  For the language-model-ish tasks y is just x shifted; for S5 it is the
running group product, which is NOT a shift -- hence the explicit y array.

Data seeds follow App. J.4: formal tasks use 20260914 / 20260915 / 20260916 for train / val / test,
addition uses 42.
"""
import itertools
import numpy as np

# ---------------------------------------------------------------------------
# One shared 277-slot vocabulary (App. J.2: "vocabulary capacity 277").
# ---------------------------------------------------------------------------
PAD, BOS, EOS, EQ, PLUS, MINUS, TIMES, LPAR, RPAR = range(9)
DIGIT0 = 9                      # digits 0-9 -> 9..18   (mod-5 uses 9..13)
PERM0 = 19                      # the 120 elements of S5 -> 19..138
VOCAB = 277

TRAIN_SEED_FORMAL, VAL_SEED_FORMAL, TEST_SEED_FORMAL = 20260914, 20260915, 20260916
ADD_SEED = 42

TASKS = ['addition', 'parity', 'mod5_flat', 'mod5_brackets', 's5_swaps', 's5_standard']

# Chance / uniform-prediction reference for each task (Fig. 2's dotted lines).
CHANCE = {'addition': None, 'parity': 50.0, 'mod5_flat': 20.0, 'mod5_brackets': 20.0,
          's5_swaps': 100.0 / 120, 's5_standard': 100.0 / 120}


# ===========================================================================
# S5 -- the symmetric group on five letters, built explicitly.
# ===========================================================================
PERMS = [tuple(p) for p in itertools.permutations(range(5))]        # 120, index = id
PERM_ID = {p: i for i, p in enumerate(PERMS)}
IDENTITY = PERM_ID[(0, 1, 2, 3, 4)]


def _compose(a, b):
    """(a o b)(i) = a[b[i]]."""
    return tuple(a[b[i]] for i in range(5))


def _transposition(i, j):
    return tuple(j if x == i else i if x == j else x for x in range(5))


# Cayley table: CAYLEY[i, j] = index of PERMS[i] o PERMS[j].  Composition becomes a gather.
CAYLEY = np.array([[PERM_ID[_compose(PERMS[i], PERMS[j])] for j in range(120)]
                   for i in range(120)], dtype=np.int64)

# Sec. 3.1: "the swaps variant uses the identity and the ten single transpositions"
TRANSPOSITIONS = [IDENTITY] + [PERM_ID[_transposition(i, j)]
                               for i in range(5) for j in range(i + 1, 5)]
assert len(TRANSPOSITIONS) == 11


# ===========================================================================
# Addition (Sec. 3.1)
#   "randomly sample 1-8-digit operands and serialize their digits in reverse order"
#   scoring: "teacher-forced answer-token accuracy [...] includes answer formatting and EOS
#             while excluding prompt and padding positions"
# ===========================================================================
def add_seq_len(nd):
    """Worst-case sequence length for operands of at most nd digits."""
    return 1 + nd + 1 + nd + 1 + (nd + 1) + 1        # BOS a + b = sum EOS


def _rand_with_digits(rng, nd):
    """Build the integer digit-by-digit.  numpy int64 tops out near 9.2e18, and the App. J.5 test
    grid goes to 32-digit operands, so this must stay in python's arbitrary-precision ints."""
    if nd == 1:
        return int(rng.integers(0, 10))
    ds = rng.integers(0, 10, size=nd)
    ds[0] = rng.integers(1, 10)                      # no leading zero
    return int(''.join(map(str, ds.tolist())))


def _rev_digits(v):
    return [DIGIT0 + int(c) for c in str(v)[::-1]]


def gen_addition(n, rng, digits=None, max_digits=8):
    L = add_seq_len(max_digits if digits is None else digits)
    x = np.full((n, L), PAD, dtype=np.int64)
    m = np.zeros((n, L), dtype=bool)
    for r in range(n):
        if digits is None:
            na, nb = rng.integers(1, max_digits + 1, size=2)
        else:
            na = nb = digits                       # test grid: both operands exactly this wide
        a = _rand_with_digits(rng, int(na))
        b = _rand_with_digits(rng, int(nb))
        toks = [BOS] + _rev_digits(a) + [PLUS] + _rev_digits(b) + [EQ] + _rev_digits(a + b) + [EOS]
        x[r, :len(toks)] = toks
        # answer tokens = everything strictly after '=' , including EOS.
        eq = toks.index(EQ)
        m[r, eq:len(toks) - 1] = True              # prediction at position i targets token i+1
    y = np.concatenate([x[:, 1:], np.full((n, 1), PAD, dtype=np.int64)], axis=1)
    return x, y, m


# ===========================================================================
# Parity (Sec. 3.1)
#   "train on binary strings of lengths 3-40 and supervise the final parity bit"
#   sequence = BOS b1..bn '='  ; the prediction made after reading '=' is the parity bit.
#   App. J.5: reported lengths "exclude BOS and any final equals sign", so length == n.
# ===========================================================================
def gen_parity(n, rng, length=None, lo=3, hi=40):
    L = (hi if length is None else length) + 2
    x = np.full((n, L), PAD, dtype=np.int64)
    y = np.full((n, L), PAD, dtype=np.int64)
    m = np.zeros((n, L), dtype=bool)
    lens = rng.integers(lo, hi + 1, size=n) if length is None else np.full(n, length)
    for r in range(n):
        k = int(lens[r])
        bits = rng.integers(0, 2, size=k)
        x[r, 0] = BOS
        x[r, 1:1 + k] = DIGIT0 + bits
        x[r, 1 + k] = EQ
        y[r, 1 + k] = DIGIT0 + int(bits.sum() % 2)
        m[r, 1 + k] = True
    return x, y, m


# ===========================================================================
# Modular arithmetic mod 5 (Sec. 3.1)
# ===========================================================================
_OPS = [PLUS, MINUS, TIMES]


def _apply(op, a, b):
    if op == PLUS:  return (a + b) % 5
    if op == MINUS: return (a - b) % 5
    return (a * b) % 5


def _eval_flat(vals, ops):
    """Multiplication binds tighter than + and -, then left to right.  Sec 3.1: 'The flat variant
    respects multiplication precedence'."""
    v, o = [vals[0]], []
    for op, b in zip(ops, vals[1:]):
        if op == TIMES:
            v[-1] = (v[-1] * b) % 5
        else:
            o.append(op); v.append(b)
    acc = v[0]
    for op, b in zip(o, v[1:]):
        acc = _apply(op, acc, b)
    return acc


def gen_mod5_flat(n, rng, length=None, lo=3, hi=39):
    """Flat expression: operands and binary operators alternate, so the token length is odd.
    `length` is the ACTUAL expression length (odd). Sequence = BOS <expr> '=' ."""
    L = (hi if length is None else length) + 2
    x = np.full((n, L), PAD, dtype=np.int64)
    y = np.full((n, L), PAD, dtype=np.int64)
    m = np.zeros((n, L), dtype=bool)
    for r in range(n):
        ell = (int(rng.integers(lo // 2, hi // 2 + 1)) * 2 + 1) if length is None else length
        k = (ell + 1) // 2                                             # number of operands
        vals = rng.integers(0, 5, size=k).tolist()
        ops = [int(_OPS[i]) for i in rng.integers(0, 3, size=max(k - 1, 0))]
        toks = [BOS]
        for i, v in enumerate(vals):
            if i: toks.append(ops[i - 1])
            toks.append(DIGIT0 + v)
        toks.append(EQ)
        x[r, :len(toks)] = toks
        y[r, len(toks) - 1] = DIGIT0 + _eval_flat(vals, ops)
        m[r, len(toks) - 1] = True
    return x, y, m


def _rand_tree(rng, k):
    """Random binary expression tree with k leaves, as (value, tokens), fully parenthesized."""
    if k == 1:
        v = int(rng.integers(0, 5))
        return v, [DIGIT0 + v]
    kl = int(rng.integers(1, k))
    lv, lt = _rand_tree(rng, kl)
    rv, rt = _rand_tree(rng, k - kl)
    op = int(_OPS[rng.integers(0, 3)])
    return _apply(op, lv, rv), [LPAR] + lt + [op] + rt + [RPAR]


def gen_mod5_brackets(n, rng, length=None, lo=3, hi=39):
    """Fully-parenthesized expression trees with the outermost parentheses dropped.

    With k leaves the serialized length is exactly 4k-5 (k>=2), so bracketed lengths live on the
    lattice {3, 7, 11, ...} == 3 (mod 4).  Training samples k directly so the length distribution
    is honest; the App. J.5 test grid (31, 39, 47, 63, 95, 127, 191, 255) is entirely on this
    lattice and is hit exactly.
    """
    L = (hi if length is None else length) + 2
    x = np.full((n, L), PAD, dtype=np.int64)
    y = np.full((n, L), PAD, dtype=np.int64)
    m = np.zeros((n, L), dtype=bool)
    k_hi = (hi + 5) // 4
    for r in range(n):
        k = int(rng.integers(2, k_hi + 1)) if length is None else (length + 5) // 4
        v, toks = _rand_tree(rng, k)
        toks = [BOS] + toks[1:-1] + [EQ]                  # drop outermost parens
        assert len(toks) <= L, (len(toks), L)
        x[r, :len(toks)] = toks
        y[r, len(toks) - 1] = DIGIT0 + v
        m[r, len(toks) - 1] = True
    return x, y, m


# ===========================================================================
# S5 state tracking (Sec. 3.1, following Grazzi et al. 2024)
#   "compose 32 permutations and supervise the running product after each input"
#   sequence = BOS p1 .. pn ; target at position i is p1 o ... o p_{i+1}.
# ===========================================================================
def gen_s5(n, rng, length=32, variant='standard'):
    alphabet = np.array(TRANSPOSITIONS if variant == 'swaps' else np.arange(120))
    idx = alphabet[rng.integers(0, len(alphabet), size=(n, length))]
    state = np.full(n, IDENTITY, dtype=np.int64)
    L = length + 1
    x = np.full((n, L), PAD, dtype=np.int64)
    y = np.full((n, L), PAD, dtype=np.int64)
    m = np.zeros((n, L), dtype=bool)
    x[:, 0] = BOS
    for t in range(length):
        state = CAYLEY[state, idx[:, t]]                       # running product
        x[:, t + 1] = PERM0 + idx[:, t]
        y[:, t] = PERM0 + state                                # read x[:,t] -> predict state_{t+1}
        m[:, t] = True
    return x, y, m


# ===========================================================================
# Dispatch
# ===========================================================================
def make(task, n, seed, split='train', length=None):
    rng = np.random.default_rng(seed)
    if task == 'addition':       return gen_addition(n, rng, digits=length)
    if task == 'parity':         return gen_parity(n, rng, length=length)
    if task == 'mod5_flat':      return gen_mod5_flat(n, rng, length=length)
    if task == 'mod5_brackets':  return gen_mod5_brackets(n, rng, length=length)
    if task == 's5_swaps':       return gen_s5(n, rng, length=length or 32, variant='swaps')
    if task == 's5_standard':    return gen_s5(n, rng, length=length or 32, variant='standard')
    raise KeyError(task)


# In-distribution validation lengths (Sec. 3.1).
VAL_LENGTHS = {
    'addition': [None],                 # mixed 1-8 digits, same range as training
    'parity': [32, 36, 40],
    'mod5_flat': [33, 35, 39],
    'mod5_brackets': [31, 35, 39],      # requested 32/36/40 -> actual, on the 3-mod-4 lattice
    's5_swaps': [32],
    's5_standard': [32],
}

# Length-generalization grids (App. J.5).
TEST_LENGTHS = {
    'addition': [9, 10, 11, 12, 14, 16, 32],
    'parity': [32, 40, 48, 64, 96, 128, 192, 256],
    'mod5_flat': [31, 39, 47, 63, 95, 127, 191, 255],
    'mod5_brackets': [31, 39, 47, 63, 95, 127, 191, 255],
    's5_swaps': [32, 48, 64, 96, 128, 192, 256, 512],
    's5_standard': [32, 48, 64, 96, 128, 192, 256, 512],
}
TRAINED_UPTO = {'addition': 8, 'parity': 40, 'mod5_flat': 39,
                'mod5_brackets': 39, 's5_swaps': 32, 's5_standard': 32}
