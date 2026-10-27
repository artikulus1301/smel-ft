"""Functional forward of one MoE decoder layer on a packed stream of sequences, and the layer-major runner that
executes a whole optimizer step layer by layer.

  * qwen3_5_moe: Gated DeltaNet or gated full attention, then experts + gated shared expert.
  * gemma4:      sliding-window or global attention (K = V, partial RoPE), then a dense MLP and the experts in
                 parallel, each with its own norms, and a per-layer output scalar."""
import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model_io import Dims
from .moe import LayerExperts, mlp_lora, moe_forward, route_gemma4, routed_experts

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
    # gemma4: RoPE of the global layers and the sliding-window mask (None = plain causal is enough)
    cos_g: torch.Tensor | None = None
    sin_g: torch.Tensor | None = None
    sliding_mask: torch.Tensor | None = None

    @property
    def T(self):
        return int(self.cu_seqlens[-1])


def _cos_sin(pos, inv_freq, dtype):
    freqs = pos.float()[:, None] * inv_freq[None]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def make_ctx(lens, d: Dims, device, dtype) -> SeqCtx:
    lens = [int(x) for x in lens]
    cu = torch.tensor([0] + lens, device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    pos = torch.cat([torch.arange(n, device=device) for n in lens])
    inv_freq = 1.0 / (d.rope_theta ** (torch.arange(0, d.rotary_dim, 2, dtype=torch.float32, device=device)
                                       / d.rotary_dim))
    cos, sin = _cos_sin(pos, inv_freq, dtype)
    mask = None
    T = int(cu[-1])
    if len(lens) > 1:
        seg = torch.repeat_interleave(torch.arange(len(lens), device=device), torch.tensor(lens, device=device))
        causal = torch.ones(T, T, dtype=torch.bool, device=device).tril_()
        mask = causal & (seg[:, None] == seg[None, :])
    ctx = SeqCtx(lens, cu, pos, cos, sin, mask)
    if d.arch == "gemma4":
        # global layers: "proportional" RoPE - the first global_rotary_dim/2 frequencies use the full head dim in
        # the exponent, the rest are 0 (cos 1, sin 0: those dims are not rotated)
        ghd, rot = d.global_head_dim, d.global_rotary_dim
        inv_g = 1.0 / (d.global_rope_theta ** (torch.arange(0, rot, 2, dtype=torch.float32, device=device) / ghd))
        inv_g = torch.cat([inv_g, torch.zeros(ghd // 2 - rot // 2, device=device)])
        ctx.cos_g, ctx.sin_g = _cos_sin(pos, inv_g, dtype)
        if max(lens) > d.sliding_window:   # the window only matters inside sequences longer than it
            i = torch.arange(T, device=device)
            win = (i[:, None] - i[None, :]) < d.sliding_window
            base = mask if mask is not None else torch.ones(T, T, dtype=torch.bool, device=device).tril_()
            ctx.sliding_mask = base & win
    return ctx


# ----------------------------------------------------------------------------------------------- primitives
def rmsnorm(x, w, eps, plus_one=True):
    """RMSNorm in fp32: x * (1 + w) (Qwen3.5) or x * w (Gemma 4, plus_one=False); w=None: no scale."""
    xf = x.float()
    if plus_one:
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return (xf * (1.0 + w.float())).type_as(x)
    xf = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    return (xf if w is None else xf * w.float()).type_as(x)


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


def attention_gemma4(x, W, d: Dims, ctx: SeqCtx, is_global: bool):
    T = x.shape[0]
    p = "self_attn."
    hd = d.global_head_dim if is_global else d.head_dim
    n_kv = d.global_n_kv if is_global else d.n_kv
    cos, sin = (ctx.cos_g, ctx.sin_g) if is_global else (ctx.cos, ctx.sin)
    q = rmsnorm(F.linear(x, W[p + "q_proj.weight"]).view(T, d.n_heads, hd), W[p + "q_norm.weight"], d.eps, False)
    k = F.linear(x, W[p + "k_proj.weight"]).view(T, n_kv, hd)
    v = k if (is_global and d.k_eq_v) else F.linear(x, W[p + "v_proj.weight"]).view(T, n_kv, hd)
    k = rmsnorm(k, W[p + "k_norm.weight"], d.eps, False)
    v = rmsnorm(v, None, d.eps, False)
    q = _rope(q.transpose(0, 1).unsqueeze(0), cos, sin)
    k = _rope(k.transpose(0, 1).unsqueeze(0), cos, sin)
    v = v.transpose(0, 1).unsqueeze(0)
    rep = d.n_heads // n_kv
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    mask = ctx.attn_mask if is_global else (ctx.sliding_mask if ctx.sliding_mask is not None else ctx.attn_mask)
    if mask is None:
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
    else:
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
    o = o.squeeze(0).transpose(0, 1).reshape(T, -1)
    return F.linear(o, W[p + "o_proj.weight"])


def layer_forward_gemma4(h, LW, d: Dims, ctx: SeqCtx, lora_layer=None, stats=None):
    W = LW.W
    x = rmsnorm(h, W["input_layernorm.weight"], d.eps, False)
    x = attention_gemma4(x, W, d, ctx, LW.ltype == "full_attention")
    h = h + rmsnorm(x, W["post_attention_layernorm.weight"], d.eps, False)
    lora_sh = lora_layer.shared if lora_layer is not None else None
    lora_ex = lora_layer.experts if lora_layer is not None else None
    # dense MLP (gets the "shared" LoRA) and the experts read the same residual through different norms
    m = mlp_lora(rmsnorm(h, W["pre_feedforward_layernorm.weight"], d.eps, False), W["mlp.gate_proj.weight"],
                 W["mlp.up_proj.weight"], W["mlp.down_proj.weight"], lora_sh, d.act)
    m = rmsnorm(m, W["post_feedforward_layernorm_1.weight"], d.eps, False)
    probs, topv, topi = route_gemma4(h, W, d.top_k, d.eps)
    if stats is not None:
        stats(probs, topv, topi)
    e = routed_experts(rmsnorm(h, W["pre_feedforward_layernorm_2.weight"], d.eps, False), topv, topi, LW.experts,
                       lora_ex)
    e = rmsnorm(e, W["post_feedforward_layernorm_2.weight"], d.eps, False)
    h = h + rmsnorm(m + e, W["post_feedforward_layernorm.weight"], d.eps, False)
    return h * W["layer_scalar"]


class LayerWeights:
    """Device tensors of one decoder layer: dense part (dict, names relative to the layer) + routed experts."""

    def __init__(self, idx: int, ltype: str, W: dict, experts: LayerExperts):
        self.idx, self.ltype, self.W, self.experts = idx, ltype, W, experts


def layer_forward(h, LW: LayerWeights, d: Dims, ctx: SeqCtx, lora_layer=None, stats=None):
    if d.arch == "gemma4":
        return layer_forward_gemma4(h, LW, d, ctx, lora_layer, stats)
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
                    hn = rmsnorm(h[rows], self.final_norm, self.d.eps, self.d.norm_plus_one)
                    loss = chunked_ce(hn, self.lm_head, t[rows], denom)
                    loss.backward()
                grads.append(h.grad)
            else:
                with torch.no_grad():
                    hn = rmsnorm(h[rows], self.final_norm, self.d.eps, self.d.norm_plus_one)
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
