"""Gated merge + recurrent decoder D_phi.  Paper Sec. 2.5, Eq. (2.9)-(2.16); App. I for chunking."""
import torch
import torch.nn as nn

from .layers import RMSNorm, SwiGLU, attend, swa_mask


class GatedMerge(nn.Module):
    """Eq. (2.9)-(2.11):

        r_{t-1} = RMSNorm_s(s_{t-1})
        g_t     = sigma( W_g [e_t ; r_{t-1}] + b_g )        W_g in R^{d x 2d}
        u_t     = e_t + alpha * g_t (*) W_s r_{t-1}          W_s in R^{d x d}

    App. I.2-I.4 (RLT-2) is the same object with the state broadcast across a chunk:

        r_{k-1} = RMSNorm_s(h_{k-1}),  v_{k-1} = W_s r_{k-1}     <- once per chunk
        g_t     = sigma( W_g [e_t ; r_{k-1}] + b_g )             <- per token
        z^0_t   = e_t + alpha * g_t (*) v_{k-1}

    Passing s_prev with a length-1 time axis gives the chunked form by broadcasting; passing a
    chunk of length 1 gives Eq. (2.11). One module, both variants, no branching.

    Parameters: d (norm) + 2d^2 + d (W_g, b_g) + d^2 (W_s) = 3d^2 + 2d = 787,456 at d=512.
    """

    def __init__(self, cfg):
        super().__init__()
        self.alpha = cfg.alpha
        self.norm_s = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.w_g = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=True)
        self.w_s = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, e, s_prev):
        """e: (B,n,d)   s_prev: (B,1,d)   ->  u: (B,n,d)"""
        r = self.norm_s(s_prev)                                   # (B,1,d)
        v = self.w_s(r)                                           # (B,1,d)  once per chunk
        g = torch.sigmoid(self.w_g(torch.cat([e, r.expand_as(e)], dim=-1)))
        return e + self.alpha * g * v


class DecoderLayer(nn.Module):
    """Eq. (2.12)-(2.16).  Sublayer order: causal SWA -> encoder-memory cross-attn -> FFN.

    Sec. 2.5: "Alternative sublayer orders define different variants and must be used consistently
    in all execution modes."  We use the published order everywhere.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        # --- causal SWA sublayer -------------------------------------------------
        self.norm_S = RMSNorm(d, cfg.norm_eps)          # RMSNorm_{S,l}, Eq. (2.13)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        # --- encoder-memory cross-attention sublayer ----------------------------
        # App. A.1: "Decoder cross-attention has separate query/output projections and the memory
        # projections of Equation (2.2)."  So: query map + output projection only.
        self.norm_M = RMSNorm(d, cfg.norm_eps)
        self.q_m = nn.Linear(d, d, bias=False)
        self.o_m = nn.Linear(d, d, bias=False)
        # --- FFN ------------------------------------------------------------------
        self.norm_D = RMSNorm(d, cfg.norm_eps)          # RMSNorm_{D,l}, Eq. (2.16)
        self.ffn = SwiGLU(d, cfg.d_ff)

    def forward(self, z, pos, cache, mem_k, mem_v, rope, scale, mem_mask, swa_m):
        """z: (B,n,d) chunk input.  pos: (n,) global positions (0-indexed).
        cache: (k,v) each (B,c,d) with c <= W-1, holding the immediately preceding positions,
               or None.
        Returns z_out (B,n,d) and the new cache.
        """
        cfg = self.cfg
        B, n, d = z.shape
        H, Dh = cfg.n_heads, cfg.head_dim

        # ---- Eq. (2.12)-(2.13): query/key maps include normalization, projection, RoPE ----
        h = self.norm_S(z)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = rope(q.view(B, n, H, Dh), pos).reshape(B, n, d)
        k = rope(k.view(B, n, H, Dh), pos).reshape(B, n, d)

        # "current KV is formed before SWA, using the layer input, so current-position attention
        #  introduces no circular dependency" (Sec. 2.5)
        if cache is not None and cache[0] is not None:
            K = torch.cat([cache[0], k], dim=1)
            V = torch.cat([cache[1], v], dim=1)
        else:
            K, V = k, v

        # ---- Eq. (2.14): b = z + Attn^D(q, SWA window) ----
        z = z + self.o_proj(attend(q, K, V, H, scale, swa_m))

        # ---- Eq. (2.15): a = b + Attn^M(P^M_Q(b, t), M_{<=t}) ----
        if mem_k is not None:
            qm = rope(self.q_m(self.norm_M(z)).view(B, n, H, Dh), pos).reshape(B, n, d)
            z = z + self.o_m(attend(qm, mem_k, mem_v, H, scale, mem_mask))

        # ---- Eq. (2.16): z^l = a + FFN(RMSNorm_D(a)) ----
        z = z + self.ffn(self.norm_D(z))

        # ---- cache update: "retain positions max(1, t-W+2), ..., t" (i.e. the last W-1) ----
        # NOTE (bug found by scripts/gates.py, gate C): the start index MUST be clamped at 0.
        # Writing K[:, K.shape[1] - w:] looks right but during warm-up K is SHORTER than W-1, so
        # that index goes negative and python silently slices from the END of the tensor instead.
        # The cache then froze at a couple of entries forever and the sliding window was never the
        # width the config asked for -- which trains perfectly happily and is invisible in a loss
        # curve. See REPORT.md "What broke".
        w = cfg.window - 1
        if w > 0:
            start = max(0, K.shape[1] - w)
            new_cache = (K[:, start:], V[:, start:])
        else:
            new_cache = (None, None)
        return z, new_cache


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.L_dec)])

    def forward(self, z, pos, caches, mem_ks, mem_vs, rope, scale, mem_mask, swa_m):
        new_caches = []
        for l, layer in enumerate(self.layers):
            g = self.cfg.group_of_layer(l)
            mk = mem_ks[g] if mem_ks is not None else None
            mv = mem_vs[g] if mem_vs is not None else None
            z, c = layer(z, pos, caches[l] if caches is not None else None,
                         mk, mv, rope, scale, mem_mask, swa_m)
            new_caches.append(c)
        return z, new_caches
