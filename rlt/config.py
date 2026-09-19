"""Model + run configuration for the RLT-1 reproduction.

Paper: Zhang, Feng, Qin. "Recurrent Looped Transformer" (Sept 2026).
All defaults are the paper's Section 3.1 values unless the field docstring says otherwise.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class RLTConfig:
    # -- vocabulary -------------------------------------------------------
    vocab_size: int = 277          # App. J.2: "vocabulary capacity 277"

    # -- width (Sec. 3.1) -------------------------------------------------
    d_model: int = 512             # "width 512"
    n_heads: int = 4               # "four attention heads"
    d_ff: int = 1365               # "FFN width 1,365" == round(8/3 * 512); => SwiGLU

    # -- depth split ------------------------------------------------------
    # Sec 3.1: "untied RLT-1 splits 4+4, 5+3, 6+2, 7+1, and 8+0"
    L_enc: int = 4                 # L_E  causal encoder layers (parallel over positions)
    L_dec: int = 4                 # L_D  recurrent decoder layers (sequential over positions)

    # -- RLT-1 specifics --------------------------------------------------
    window: int = 8                # W, "an SWA window of eight". W includes the CURRENT token
                                   # (Sec 2.1: "The window size W >= 1 includes the current token,
                                   #  so each layer retains at most W-1 past positions").
    n_mem_groups: int = 1          # G, "one shared encoder-memory group". G=1 => every decoder
                                   # layer reads the same store (Sec 2.2).
    alpha: float = 0.1             # feedback scale in Eq. (2.11); "The experiments use alpha = 0.1"

    # -- architecture variant ---------------------------------------------
    feedback: bool = True          # True  -> RLT-1 / RLT-2  (gated merge + learned s_star)
                                   # False -> RLT-0 control (App. F.2): removes the learned initial
                                   #          state, state normalization, merge gate, and feedback
                                   #          projection; decoder input is e_t directly.
    chunk: int = 1                 # B, App. I. B=1 is exactly RLT-1 (App. I.1: "With B = 1, the
                                   # transition reduces to RLT-1 at identical weights"). B>1 is
                                   # RLT-2: the feedback state is held fixed inside a chunk so the
                                   # chunk's positions run in parallel. B=0 means "one chunk = whole
                                   # sequence" (used only by RLT-0, which has no feedback at all).
    tie_enc_dec: bool = False      # App. A.1. Sec 3.1 evaluates the UNTIED models.

    # -- positional / norm ------------------------------------------------
    rope_base: float = 10000.0     # "keys include positional transformations" (Fig. 6)
    norm_eps: float = 1e-6
    tie_readout: bool = True       # derived from Table 4 (see Part 4)

    # -- width-mu-P (Sec. 3.1: "Both architectures use the same width-muP recipe") ----
    mup_base_width: int = 512      # anchor = the paper's own width, so the paper's LR transfers
    init_std: float = 0.02         # sigma at the base width

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert self.window >= 1
        assert self.L_dec == 0 or self.n_mem_groups in (1, self.L_dec), \
            "Sec 2.2: G=1 shares memory across layers, G=L_D gives each layer its own projections"

    # ---- derived ----
    @property
    def head_dim(self):        return self.d_model // self.n_heads
    @property
    def mup_mult(self):        return self.mup_base_width / self.d_model   # base/width
    @property
    def n_layers_total(self):  return self.L_enc + self.L_dec
    @property
    def has_memory(self):
        """RLT-1 8+0 has no decoder blocks, so nothing reads encoder memory and the memory
        projection of Eq. (2.2) is not instantiated. Table 4 confirms this: 8+0 is 26.10M, which
        is 7+1 minus one decoder block plus one encoder block MINUS the 0.525M memory projection."""
        return self.L_dec > 0

    def group_of_layer(self, l):
        """g(l) in Eq. (2.2): which memory group decoder layer l reads."""
        return 0 if self.n_mem_groups == 1 else l

    def to_dict(self):  return asdict(self)


@dataclass
class GPTConfig:
    """The decoder-only Transformer baseline ("Transformer 8" / "GPT 8" in the paper).

    Same width / heads / FFN / vocab as RLT-1; full causal attention (no sliding window), which
    is the standard and strictly more expressive choice, so the baseline is not handicapped.
    """
    vocab_size: int = 277
    d_model: int = 512
    n_heads: int = 4
    d_ff: int = 1365
    n_layers: int = 8
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    tie_readout: bool = True
    mup_base_width: int = 512
    init_std: float = 0.02

    @property
    def head_dim(self):  return self.d_model // self.n_heads
    @property
    def mup_mult(self):  return self.mup_base_width / self.d_model

    def to_dict(self):  return asdict(self)


# ---------------------------------------------------------------------------
# The five depth splits of Sec. 3.1 plus the baseline, by name.
# ---------------------------------------------------------------------------
SPLITS = {
    'rlt1_4+4': (4, 4),
    'rlt1_5+3': (5, 3),
    'rlt1_6+2': (6, 2),
    'rlt1_7+1': (7, 1),
    'rlt1_8+0': (8, 0),
}
ARCHS = list(SPLITS) + ['gpt_8']
SEEDS = [42, 43, 44]          # Sec. 3.1: "initialization seeds 42, 43, and 44"
