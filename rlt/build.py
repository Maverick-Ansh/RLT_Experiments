"""Model factory: name -> module.  Keeps the six architectures of Sec. 3.1 in one place."""
from .config import RLTConfig, GPTConfig, SPLITS
from .model import RLT, GPT


def build(arch, vocab_size, d_model=512, d_ff=None, n_heads=4, **kw):
    """arch in {'rlt1_4+4', ..., 'rlt1_8+0', 'gpt_8', 'rlt0_4+4', ...}.

    d_ff defaults to round(8/3 * d_model), matching the paper's 1365 at d=512.
    """
    if d_ff is None:
        d_ff = int(round(8 * d_model / 3))
    common = dict(vocab_size=vocab_size, d_model=d_model, d_ff=d_ff, n_heads=n_heads)
    if arch.startswith('gpt'):
        n_layers = int(arch.split('_')[1])
        return GPT(GPTConfig(n_layers=n_layers, **common, **kw))
    feedback = not arch.startswith('rlt0')
    split = arch.split('_', 1)[1]
    le, ld = (int(v) for v in split.split('+'))
    cfg = RLTConfig(L_enc=le, L_dec=ld, feedback=feedback, **common, **kw)
    if not feedback:
        # App. F.2: RLT-0's known positions "run together within each decoder layer", i.e. the
        # whole sequence is one chunk. chunk=0 encodes that.
        cfg.chunk = 0
    return RLT(cfg)
