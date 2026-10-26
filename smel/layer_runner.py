"""Functional forward of one Qwen3.5/3.6-MoE decoder layer (Gated DeltaNet or gated full attention + MoE) on a
packed stream of sequences, and the layer-major runner that executes a whole optimizer step layer by layer."""
import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model_io import Dims
from .moe import LayerExperts, moe_forward

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk_gdr
except Exception:  # pragma: no cover
    _fla_chunk_gdr = None


def _torch_reference_gdr():
    """Pure-torch chunked gated delta rule from transformers, without the hub-kernel decorator (which would swap it
    for FLA when FLA is installed)."""
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M
    src = inspect.getsource(M).split("def torch_chunk_gated_delta_rule(")[1].split("\n@use_kernel_func_from_hub")[0]
    ns = dict(M.__dict__)
    exec("def torch_chunk_gated_delta_rule(" + src, ns)
    return ns["torch_chunk_gated_delta_rule"]


_TORCH_GDR = None
USE_FLA = _fla_chunk_gdr is not None


def gated_delta_rule(q, k, v, g, beta, cu_seqlens, lens):
    global _TORCH_GDR
    if USE_FLA and q.is_cuda:
        o, _ = _fla_chunk_gdr(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True,
                              cu_seqlens=cu_seqlens if len(lens) > 1 else None)
        return o
    if _TORCH_GDR is None:
        _TORCH_GDR = _torch_reference_gdr()
    outs, s = [], 0
    for n in lens:
        o, _ = _TORCH_GDR(q[:, s:s + n], k[:, s:s + n], v[:, s:s + n], g[:, s:s + n], beta[:, s:s + n],
                          use_qk_l2norm_in_kernel=True)
        outs.append(o)
        s += n
    return torch.cat(outs, 1)


# ----------------------------------------------------------------------------------------------- sequence context
@dataclass
class SeqCtx:
    lens: list
    cu_seqlens: torch.Tensor  # int32 [n+1]
    pos: torch.Tensor  # position inside its sequence, long [T]
    cos: torch.Tensor  # [T, rotary_dim]
    sin: torch.Tensor
    attn_mask: torch.Tensor | None  # bool [T, T] (True = attend) or None for one sequence

    @property
    def T(self):
        return int(self.cu_seqlens[-1])


def make_ctx(lens, d: Dims, device, dtype) -> SeqCtx:
    lens = [int(x) for x in lens]
    cu = torch.tensor([0] + lens, device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    pos = torch.cat([torch.arange(n, device=device) for n in lens])
    inv_freq = 1.0 / (d.rope_theta ** (torch.arange(0, d.rotary_dim, 2, dtype=torch.float32, device=device)
                                       / d.rotary_dim))
    freqs = pos.float()[:, None] * inv_freq[None]
    emb = torch.cat([freqs, freqs], dim=-1)
    mask = None
    if len(lens) > 1:
        seg = torch.repeat_interleave(torch.arange(len(lens), device=device), torch.tensor(lens, device=device))
        T = seg.numel()
        causal = torch.ones(T, T, dtype=torch.bool, device=device).tril_()
        mask = causal & (seg[:, None] == seg[None, :])
    return SeqCtx(lens, cu, pos, emb.cos().to(dtype), emb.sin().to(dtype), mask)


# ----------------------------------------------------------------------------------------------- primitives
def rmsnorm(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * (1.0 + w.float())).type_as(x)


def gated_rmsnorm(x, z, w, eps):
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = w * xf.to(dt)
    return (y * F.silu(z.float())).to(dt)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rope(x, cos, sin):
    # x [1, H, T, D]; cos/sin [T, R] applied to the first R dims
    R = cos.shape[-1]
    xr, xp = x[..., :R], x[..., R:]
    xr = xr * cos + _rotate_half(xr) * sin
    return torch.cat([xr, xp], dim=-1)


def causal_conv_silu(x, w, pos):
    """Depthwise causal conv1d over a packed stream; taps never cross sequence starts. x [T, C], w [C, 1, K]."""
    K = w.shape[-1]
    w = w.reshape(w.shape[0], K).float()
    xf = x.float()
    out = xf * w[:, K - 1]
    for s in range(1, K):
        sh = F.pad(xf[:-s], (0, 0, s, 0)) if s < xf.shape[0] else torch.zeros_like(xf)
        sh = sh * (pos >= s).unsqueeze(1)
        out = out + sh * w[:, K - 1 - s]
    return F.silu(out).to(x.dtype)


def attention(x, W, d: Dims, ctx: SeqCtx):
    T = x.shape[0]
    p = "self_attn."
    qg = F.linear(x, W[p + "q_proj.weight"]).view(T, d.n_heads, 2 * d.head_dim)
    q, gate = qg.chunk(2, dim=-1)
    gate = gate.reshape(T, -1)
    q = rmsnorm(q, W[p + "q_norm.weight"], d.eps).transpose(0, 1).unsqueeze(0)
    k = rmsnorm(F.linear(x, W[p + "k_proj.weight"]).view(T, d.n_kv, d.head_dim), W[p + "k_norm.weight"], d.eps)
    k = k.transpose(0, 1).unsqueeze(0)
    v = F.linear(x, W[p + "v_proj.weight"]).view(T, d.n_kv, d.head_dim).transpose(0, 1).unsqueeze(0)
    q, k = _rope(q, ctx.cos, ctx.sin), _rope(k, ctx.cos, ctx.sin)
    rep = d.n_heads // d.n_kv
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    if ctx.attn_mask is None:
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=d.head_dim ** -0.5)
    else:
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx.attn_mask, scale=d.head_dim ** -0.5)
    o = o.squeeze(0).transpose(0, 1).reshape(T, -1)
    o = o * torch.sigmoid(gate)
    return F.linear(o, W[p + "o_proj.weight"])


def deltanet(x, W, d: Dims, ctx: SeqCtx):
    T = x.shape[0]
    p = "linear_attn."
    kd, vd = d.lin_k_heads * d.lin_k_dim, d.lin_v_heads * d.lin_v_dim
    qkv = F.linear(x, W[p + "in_proj_qkv.weight"])
    z = F.linear(x, W[p + "in_proj_z.weight"])
    b = F.linear(x, W[p + "in_proj_b.weight"])
    a = F.linear(x, W[p + "in_proj_a.weight"])
    qkv = causal_conv_silu(qkv, W[p + "conv1d.weight"], ctx.pos)
    q, k, v = torch.split(qkv, [kd, kd, vd], dim=-1)
    q = q.reshape(1, T, d.lin_k_heads, d.lin_k_dim)
    k = k.reshape(1, T, d.lin_k_heads, d.lin_k_dim)
    v = v.reshape(1, T, d.lin_v_heads, d.lin_v_dim)
    beta = torch.sigmoid(b).unsqueeze(0)
    g = (-W[p + "A_log"].float().exp() * F.softplus(a.float() + W[p + "dt_bias"])).unsqueeze(0)
    rep = d.lin_v_heads // d.lin_k_heads
    if rep > 1:
        q = q.repeat_interleave(rep, dim=2)
        k = k.repeat_interleave(rep, dim=2)
    o = gated_delta_rule(q, k, v, g, beta, ctx.cu_seqlens, ctx.lens)
    o = gated_rmsnorm(o.reshape(-1, d.lin_v_dim), z.reshape(-1, d.lin_v_dim), W[p + "norm.weight"], d.eps)
    return F.linear(o.reshape(T, vd), W[p + "out_proj.weight"])


class LayerWeights:
    """Device tensors of one decoder layer: dense part (dict, names relative to the layer) + routed experts."""

    def __init__(self, idx: int, ltype: str, W: dict, experts: LayerExperts):
        self.idx, self.ltype, self.W, self.experts = idx, ltype, W, experts


def layer_forward(h, LW: LayerWeights, d: Dims, ctx: SeqCtx, lora_layer=None, stats=None):
    W = LW.W
    x = rmsnorm(h, W["input_layernorm.weight"], d.eps)
    if LW.ltype == "linear_attention":
        x = deltanet(x, W, d, ctx)
    else:
        x = attention(x, W, d, ctx)
    h = h + x
    x = rmsnorm(h, W["post_attention_layernorm.weight"], d.eps)
    return h + moe_forward(x, W, LW.experts, d.top_k, lora_layer, stats)


# ----------------------------------------------------------------------------------------------- layer-major runner
class LayerMajorRunner:
    """Runs layers [first, last] of the model on a list of packed microbatches, layer by layer.

    forward: every layer is loaded once and applied to all microbatches (inputs of each layer are checkpointed).
    backward: layers are loaded again in reverse order; each microbatch is recomputed under autograd and
    back-propagated, which accumulates LoRA grads and yields the gradient w.r.t. the layer input.
    """

    def __init__(self, dims: Dims, source, layers, lora=None, final_norm=None, lm_head=None, ckpt_device=None,
                 timer=None):
        self.d, self.src, self.layers = dims, source, list(layers)
        self.lora, self.final_norm, self.lm_head = lora, final_norm, lm_head
        self.ckpt_device = ckpt_device
        self.timer = timer

    def _lora(self, l):
        return self.lora.get(l) if self.lora is not None else None

    def _get(self, i, order):
        LW = self.src.get(order[i])
        if i + 1 < len(order):
            self.src.prefetch(order[i + 1])
        return LW

    @torch.no_grad()
    def forward(self, hs, ctxs, keep_ckpt=False, stats_cb=None):
        ckpt = {}
        order = self.layers
        for i, l in enumerate(order):
            LW = self._get(i, order)
            if keep_ckpt:
                ckpt[l] = [h if self.ckpt_device is None else h.to(self.ckpt_device, non_blocking=True) for h in hs]
            for m in range(len(hs)):
                st = (lambda p, v, t, _l=l, _m=m: stats_cb(_l, _m, p, v, t)) if stats_cb else None
                hs[m] = layer_forward(hs[m], LW, self.d, ctxs[m], self._lora(l), st)
        return hs, ckpt

    def head_loss(self, hs, targets, denom, grad=True):
        """Final norm + chunked CE on supervised positions. Returns (loss_sum/denom, grads wrt hs or None)."""
        from .loss import chunked_ce
        total, grads = 0.0, []
        for h, t in zip(hs, targets):
            rows = (t != -100).nonzero().squeeze(1)
            if grad:
                h = h.detach().requires_grad_(True)
                with torch.enable_grad():
                    hn = rmsnorm(h[rows], self.final_norm, self.d.eps)
                    loss = chunked_ce(hn, self.lm_head, t[rows], denom)
                    loss.backward()
                grads.append(h.grad)
            else:
                with torch.no_grad():
                    hn = rmsnorm(h[rows], self.final_norm, self.d.eps)
                    loss = chunked_ce(hn, self.lm_head, t[rows], denom, need_grad=False)
            total += float(loss)
        return total, (grads if grad else None)

    def backward(self, ckpt, grads, ctxs):
        order = self.layers[::-1]
        for i, l in enumerate(order):
            LW = self._get(i, order)
            for m in range(len(grads)):
                x = ckpt[l][m]
                x = x.to(grads[m].device, non_blocking=True).detach().requires_grad_(True)
                with torch.enable_grad():
                    y = layer_forward(x, LW, self.d, ctxs[m], self._lora(l))
                    y.backward(grads[m])
                grads[m] = x.grad
                ckpt[l][m] = None
        return grads

    def train_step(self, hs, ctxs, targets, denom):
        hs, ckpt = self.forward(hs, ctxs, keep_ckpt=True)
        loss, grads = self.head_loss(hs, targets, denom, grad=True)
        del hs
        self.backward(ckpt, grads, ctxs)
        return loss

    @torch.no_grad()
    def eval_loss(self, hs, ctxs, targets, denom, stats_cb=None):
        hs, _ = self.forward(hs, ctxs, keep_ckpt=False, stats_cb=stats_cb)
        loss, _ = self.head_loss(hs, targets, denom, grad=False)
        return loss
