"""Primitives, written from scratch.

No nn.MultiheadAttention / nn.TransformerEncoderLayer anywhere in this repo. The only torch
helper used inside attention is F.scaled_dot_product_attention, which is a fused matmul-softmax-
matmul -- the mask and the positional transform are ours.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm.  The paper uses RMSNorm everywhere (Eq. 2.6, 2.9, 2.13, 2.16).
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        # rsqrt(mean(x^2)) in fp32 regardless of the autocast dtype: the variance of a width-d
        # vector in fp16 overflows well before d=512 matters, and this is the cheap fix.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


# ---------------------------------------------------------------------------
# Rotary positional transform.
# Fig. 6 / Eq. (2.2), (2.12), (2.13): "Keys include positional transformations", and the key map
# P_K "includes normalization, projection, and any positional transformation".
# ---------------------------------------------------------------------------
class Rotary(nn.Module):
    def __init__(self, head_dim, base=10000.0, max_len=4096):
        super().__init__()
        assert head_dim % 2 == 0
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv)                       # (max_len, head_dim/2)
        self.register_buffer('cos', freqs.cos(), persistent=False)
        self.register_buffer('sin', freqs.sin(), persistent=False)
        self.max_len = max_len

    def forward(self, x, pos):
        """x: (B, N, H, Dh)   pos: (N,) int64 global positions.   -> rotated x"""
        cos = self.cos[pos].to(x.dtype)[None, :, None, :]  # (1, N, 1, Dh/2)
        sin = self.sin[pos].to(x.dtype)[None, :, None, :]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


# ---------------------------------------------------------------------------
# Masked multi-head attention.
# ---------------------------------------------------------------------------
def attend(q, k, v, n_heads, scale, mask=None):
    """q: (B,Nq,d)  k,v: (B,Nk,d)  mask: (Nq,Nk) bool, True = may attend (or None = dense).

    Returns (B,Nq,d).  `scale` is passed explicitly so mu-P can override the usual
    1/sqrt(head_dim).
    """
    B, Nq, d = q.shape
    Nk = k.shape[1]
    H = n_heads
    Dh = d // H
    q = q.view(B, Nq, H, Dh).transpose(1, 2)
    k = k.view(B, Nk, H, Dh).transpose(1, 2)
    v = v.view(B, Nk, H, Dh).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    return o.transpose(1, 2).reshape(B, Nq, d)


def swa_mask(q_pos, k_pos, window):
    """Causal sliding-window mask.  Sec. 2.5, Eq. (2.14):
        Attn^D( q_t , { (k_j, v_j) }_{j = max(1, t-W+1)}^{t} )
    so key j is visible to query t iff  0 <= t - j <= W-1.  W counts the current token (Sec. 2.1).
    """
    d = q_pos[:, None] - k_pos[None, :]
    return (d >= 0) & (d <= window - 1)


def prefix_mask(q_pos, k_pos):
    """Encoder-memory mask.  Sec. 2.2 / Eq. (2.15): cross-attention reads M_{<=t}, i.e. key j
    visible to query t iff j <= t."""
    return k_pos[None, :] <= q_pos[:, None]


# ---------------------------------------------------------------------------
# SwiGLU feed-forward.
# Inferred from Sec. 3.1's "FFN width 1,365" at d=512: 1365 = round(8/3 * 512). A two-matrix
# GELU FFN at the usual 4d would have been 2048. Part 4 confirms the choice against Table 4.
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up   = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
# width-mu-P.
# Sec. 3.1: "Both architectures use the same width-muP recipe".
# Anchored at mup_base_width so that at width == base this is exactly standard parameterization,
# and the paper's LR (tuned at d=512) transfers to any other width.
# ---------------------------------------------------------------------------
def mup_attn_scale(head_dim, base_head_dim, enabled=True):
    """mu-P uses 1/d_head attention logits instead of 1/sqrt(d_head), normalized so that it
    coincides with 1/sqrt(d_head) at the base width."""
    if not enabled:
        return head_dim ** -0.5
    return math.sqrt(base_head_dim) / head_dim


@torch.no_grad()
def mup_init_(module, cfg, kind, n_residual_layers=1):
    """kind: 'embed' | 'hidden' | 'readout'.

    mu-P: hidden weights keep var ~ 1/fan_in, so std scales as sqrt(base/width). Input-like
    (embedding) layers keep a width-independent std.
    """
    m = cfg.mup_base_width / cfg.d_model
    if kind == 'embed':
        std = cfg.init_std
    else:
        std = cfg.init_std * math.sqrt(m)
    for name, p in module.named_parameters():
        if p.dim() >= 2:
            p.normal_(0.0, std)
            # GPT-2 style residual-branch downscaling: the *output* projection of each residual
            # sublayer is divided by sqrt(2 * n_layers) so the residual stream variance does not
            # grow with depth. Applied to all six depth splits identically.
            if name.endswith(('w_down.weight', 'o_proj.weight')):
                p.mul_(1.0 / math.sqrt(2 * max(n_residual_layers, 1)))


def mup_param_groups(model, cfg, lr, weight_decay):
    """Adam LR scales as base/width for hidden matrices; embeddings, norms, biases and the learned
    initial state keep the base LR. Weight decay applies to matrices only."""
    m = cfg.mup_base_width / cfg.d_model
    hidden, flat = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (hidden if p.dim() >= 2 and 'embed' not in name else flat).append(p)
    return [
        {'params': hidden, 'lr': lr * m,  'weight_decay': weight_decay},
        {'params': flat,   'lr': lr,      'weight_decay': 0.0},
    ]
