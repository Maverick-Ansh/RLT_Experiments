"""RLT-1 / RLT-2 / RLT-0 and the decoder-only Transformer baseline.

Sec. 2.3, Eq. (2.3)-(2.7):
    H_0 = (s_star, empty)
    u_t = Merge(e_t, s_{t-1})
    H_t = (s_t, C^D_t) = D_phi(u_t ; M_{<=t}, C^D_{t-1}, t)
    p(x_{t+1} | x_{1:t}) = softmax( W_o RMSNorm_o(s_t) )
    H_T = F_T o F_{T-1} o ... o F_1 (H_0)
"""
import math
import torch
import torch.nn as nn

from .layers import (RMSNorm, Rotary, SwiGLU, attend, swa_mask, prefix_mask,
                     mup_attn_scale, mup_init_)
from .encoder import Encoder, EncoderMemory
from .decoder import Decoder, GatedMerge


class RLT(nn.Module):
    """RLT-1 (chunk=1), RLT-2 (chunk>1) and RLT-0 (feedback=False) in one module."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.rope = Rotary(cfg.head_dim, cfg.rope_base)
        self.scale = mup_attn_scale(cfg.head_dim, cfg.mup_base_width // cfg.n_heads)

        self.encoder = Encoder(cfg)
        self.memory = EncoderMemory(cfg) if cfg.has_memory else None
        self.decoder = Decoder(cfg)

        if cfg.feedback:
            self.merge = GatedMerge(cfg)
            # Sec. 2.1: "We collect all parameters, including the learned initial state s_star".
            self.s_star = nn.Parameter(torch.zeros(1, 1, d))
        else:
            self.merge, self.s_star = None, None

        self.norm_o = RMSNorm(d, cfg.norm_eps)                    # RMSNorm_o, Eq. (2.6)
        self.readout = nn.Linear(d, cfg.vocab_size, bias=False)
        if cfg.tie_readout:
            self.readout.weight = self.embed.weight

        self._init()

    def _init(self):
        c = self.cfg
        L = max(c.n_layers_total, 1)
        mup_init_(self.embed, c, 'embed')
        for mod in [self.encoder, self.decoder]:
            mup_init_(mod, c, 'hidden', L)
        if self.memory is not None:
            mup_init_(self.memory, c, 'hidden', L)
        if self.merge is not None:
            mup_init_(self.merge, c, 'hidden', L)
            nn.init.zeros_(self.merge.w_g.bias)
            nn.init.zeros_(self.s_star)
        if not c.tie_readout:
            mup_init_(self.readout, c, 'hidden', L)

    # -- helpers -----------------------------------------------------------
    def _chunks(self, T):
        """App. I.1: a_k = (k-1)B + 1, b_k = min(kB, T).  chunk=0 means one chunk for the whole
        sequence (only legal without feedback)."""
        B = self.cfg.chunk
        if B <= 0:
            assert not self.cfg.feedback
            return [(0, T - 1)]
        return [(a, min(a + B, T) - 1) for a in range(0, T, B)]

    def encode(self, x, pos=None):
        """Eq. (2.1)-(2.2): causal encoder + global memory, both parallel over positions."""
        B, T = x.shape
        dev = x.device
        if pos is None:
            pos = torch.arange(T, device=dev)
        cm = prefix_mask(pos, pos)                                 # causal
        e, enc_cache = self.encoder(self.embed(x), pos, self.rope, self.scale, cm)
        if self.memory is not None:
            mk, mv = self.memory(e, pos, self.rope)
        else:
            mk = mv = None
        return e, mk, mv, enc_cache, pos

    def run_decoder(self, e, mk, mv, pos, s=None, caches=None, mem_pos=None):
        """The recurrence of Eq. (2.4)-(2.5) / App. I.2-I.7.  Returns y (B,T,d) and final state."""
        cfg = self.cfg
        Bn, T, d = e.shape
        dev = e.device
        if s is None and cfg.feedback:
            s = self.s_star.expand(Bn, 1, d)                       # H_0 = (s_star, empty)
        if caches is None:
            caches = [(None, None)] * cfg.L_dec
        if mem_pos is None:
            mem_pos = pos
        # Eq. (2.15) mask: query t reads encoder memory j <= t.  Built once, sliced per chunk.
        mem_mask_full = prefix_mask(pos, mem_pos) if mk is not None else None

        outs = []
        for (a, b) in self._chunks(T):
            n = b - a + 1
            e_c = e[:, a:b + 1]
            pos_c = pos[a:b + 1]
            u = self.merge(e_c, s) if cfg.feedback else e_c        # Eq. (2.4) / (I.4)
            # SWA mask.  For n == 1 every cached key is within the window by construction, so the
            # mask is all-True and omitted (this is the hot path -- it runs T times per example).
            if n == 1 or cfg.L_dec == 0:
                swa_m = None
            else:
                c_len = 0 if caches[0][0] is None else caches[0][0].shape[1]
                k_pos = torch.arange(int(pos_c[0]) - c_len, int(pos_c[-1]) + 1, device=dev)
                swa_m = swa_mask(pos_c, k_pos, cfg.window)
            mem_mask = mem_mask_full[a:b + 1] if mem_mask_full is not None else None
            z, caches = self.decoder(u, pos_c, caches, mk, mv, self.rope, self.scale,
                                     mem_mask, swa_m)
            outs.append(z)
            if cfg.feedback:
                # Eq. (2.16) s_t = z^{L_D}_t   /   Eq. (I.7) h_k = y_{kB} at a full boundary.
                # A partial final chunk still commits here; it is the last chunk of the sequence,
                # so nothing reads the state afterwards during teacher-forced training.
                s = z[:, -1:, :]
        y = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
        return y, s, caches

    # -- full teacher-forced pass (training / prefill) ---------------------
    def forward(self, x, return_state=False):
        e, mk, mv, enc_cache, pos = self.encode(x)
        y, s, caches = self.run_decoder(e, mk, mv, pos)
        logits = self.readout(self.norm_o(y)) * self.cfg.mup_mult ** 0
        if return_state:
            return logits, dict(s=s, caches=caches, enc_cache=enc_cache, mk=mk, mv=mv, pos=pos, T=x.shape[1])
        return logits

    # -- incremental execution (App. F.3) ----------------------------------
    @torch.no_grad()
    def step(self, tok, st):
        """Consume ONE new token given the state dict returned by forward(..., return_state=True).

        Eq. (2.8): (e_t, C^E_t) = E^step_theta(x_t, C^E_{t-1}); then a memory append and one
        decoder update.  Returns (logits_for_next_token, new_state).
        """
        cfg = self.cfg
        dev = tok.device
        t = st['T']
        pos = torch.tensor([t], device=dev)
        e, enc_cache = self.encoder(self.embed(tok[:, None]), pos, self.rope, self.scale, None,
                                    st['enc_cache'])
        mk, mv = st['mk'], st['mv']
        if self.memory is not None:
            k_new, v_new = self.memory(e, pos, self.rope)
            mk = [torch.cat([a, b], 1) for a, b in zip(mk, k_new)]
            mv = [torch.cat([a, b], 1) for a, b in zip(mv, v_new)]
        all_pos = torch.arange(t + 1, device=dev)
        y, s, caches = self.run_decoder(e, mk, mv, pos, s=st['s'], caches=st['caches'],
                                        mem_pos=all_pos)
        logits = self.readout(self.norm_o(y))
        return logits, dict(s=s, caches=caches, enc_cache=enc_cache, mk=mk, mv=mv,
                            pos=all_pos, T=t + 1)

    def n_params(self, unique=True):
        seen, n = set(), 0
        for p in self.parameters():
            if unique and id(p) in seen:
                continue
            seen.add(id(p)); n += p.numel()
        return n


# ---------------------------------------------------------------------------
# Baseline: decoder-only Transformer ("Transformer 8" / "GPT 8").
# ---------------------------------------------------------------------------
class GPTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.norm_attn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.norm_ffn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, pos, rope, scale, mask, cache=None):
        h = self.norm_attn(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        B, N, d = q.shape
        H, Dh = self.cfg.n_heads, self.cfg.head_dim
        q = rope(q.view(B, N, H, Dh), pos).reshape(B, N, d)
        k = rope(k.view(B, N, H, Dh), pos).reshape(B, N, d)
        if cache is not None and cache[0] is not None:
            k = torch.cat([cache[0], k], 1); v = torch.cat([cache[1], v], 1)
            mask = None if N == 1 else mask
        x = x + self.o_proj(attend(q, k, v, H, scale, mask))
        x = x + self.ffn(self.norm_ffn(x))
        return x, (k, v)


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = Rotary(cfg.head_dim, cfg.rope_base)
        self.scale = mup_attn_scale(cfg.head_dim, cfg.mup_base_width // cfg.n_heads)
        self.blocks = nn.ModuleList([GPTBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_o = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.readout = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_readout:
            self.readout.weight = self.embed.weight
        mup_init_(self.embed, cfg, 'embed')
        for b in self.blocks:
            mup_init_(b, cfg, 'hidden', cfg.n_layers)
        if not cfg.tie_readout:
            mup_init_(self.readout, cfg, 'hidden', cfg.n_layers)

    def forward(self, x, caches=None, pos=None, return_state=False):
        B, T = x.shape
        if pos is None:
            pos = torch.arange(T, device=x.device)
        mask = prefix_mask(pos, pos)
        h = self.embed(x)
        new = []
        for i, blk in enumerate(self.blocks):
            h, c = blk(h, pos, self.rope, self.scale, mask,
                       caches[i] if caches is not None else None)
            new.append(c)
        logits = self.readout(self.norm_o(h))
        return (logits, new) if return_state else logits

    def n_params(self, unique=True):
        seen, n = set(), 0
        for p in self.parameters():
            if unique and id(p) in seen:
                continue
            seen.add(id(p)); n += p.numel()
        return n
