"""Causal encoder E_theta and the encoder-derived KV memory.  Paper Sec. 2.2, Eq. (2.1)-(2.2).

Supports both execution modes the paper needs:
  * parallel over a known prefix        -- Eq. (2.1), used in training and prompt prefill
  * incremental with an encoder cache   -- Eq. (2.8), (e_t, C^E_t) = E^step_theta(x_t, C^E_{t-1})
Prop. B.1 asserts these agree; Part 5 checks it numerically.
"""
import torch
import torch.nn as nn

from .layers import RMSNorm, Rotary, SwiGLU, attend


class EncoderLayer(nn.Module):
    """One causal pre-norm block.

    Sec. 2.2: "Positions within each encoder layer can be processed together using a causal mask.
    Encoder layers remain sequential."  Nothing exotic -- the encoder's only special property is
    that it is parallel over positions.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.norm_attn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.norm_ffn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, pos, rope, scale, causal_mask, cache=None):
        h = self.norm_attn(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        B, N, d = q.shape
        H, Dh = self.cfg.n_heads, self.cfg.head_dim
        q = rope(q.view(B, N, H, Dh), pos).reshape(B, N, d)
        k = rope(k.view(B, N, H, Dh), pos).reshape(B, N, d)
        if cache is not None and cache[0] is not None:
            k = torch.cat([cache[0], k], dim=1)
            v = torch.cat([cache[1], v], dim=1)
            # a single new query at position t may attend to every cached position j <= t, so the
            # mask is vacuously all-True and is skipped.
            causal_mask = None if N == 1 else causal_mask
        x = x + self.o_proj(attend(q, k, v, H, scale, causal_mask))
        x = x + self.ffn(self.norm_ffn(x))
        return x, (k, v)


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.L_enc)])

    def forward(self, x, pos, rope, scale, causal_mask, caches=None):
        new = []
        for i, layer in enumerate(self.layers):
            x, c = layer(x, pos, rope, scale, causal_mask,
                         caches[i] if caches is not None else None)
            new.append(c)
        return x, new                              # e_{1:T}, Eq. (2.1);  C^E, Eq. (2.8)


class EncoderMemory(nn.Module):
    """Eq. (2.2).

        k^g_t = P^g_K(e_t, t)              key map = normalization + projection + RoPE
        v^g_t = W^g_V RMSNorm_E(e_t)
        M^g_{<=t} = {(k^g_j, v^g_j)}_{j=1}^{t}

    G = n_mem_groups.  Sec. 2.2: "Decoder layer l reads group g(l): G = 1 shares memory across
    layers, while G = L_D allows separate projections for each layer."

    Note there is ONE RMSNorm_E shared by all groups (it carries no g superscript in Eq. 2.2);
    only the K and V projections are per-group.  Table 4 confirms the count: for G=1 this module
    is d + 2d^2 = 524,800 parameters at d=512, which is exactly the 8+0 -> 7+1 gap.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        G = cfg.n_mem_groups
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)          # RMSNorm_E
        self.k_proj = nn.ModuleList([nn.Linear(cfg.d_model, cfg.d_model, bias=False) for _ in range(G)])
        self.v_proj = nn.ModuleList([nn.Linear(cfg.d_model, cfg.d_model, bias=False) for _ in range(G)])

    def forward(self, e, pos, rope):
        """e: (B,T,d) -> lists of G tensors, each (B,T,d).  Appending to M is a concat by caller."""
        h = self.norm(e)
        B, T, d = h.shape
        H, Dh = self.cfg.n_heads, self.cfg.head_dim
        ks, vs = [], []
        for kp, vp in zip(self.k_proj, self.v_proj):
            k = rope(kp(h).view(B, T, H, Dh), pos).reshape(B, T, d)
            ks.append(k)
            vs.append(vp(h))
        return ks, vs
