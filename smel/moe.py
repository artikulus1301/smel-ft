"""Own MoE forward: router -> dispatch -> frozen experts (bf16 or int4) / selected bf16 experts + LoRA -> combine.

Frozen experts run inside a custom autograd Function that keeps only its input for backward and re-materializes
(dequantizes) the weights group by group, so a backward pass never holds a whole layer of experts in bf16.
"""
import torch
import torch.nn.functional as F

from .lora import lora_linear
from .quant import dequant_int4, quantize_int4

try:
    _HAS_GROUPED_MM = hasattr(torch, "_grouped_mm")
except Exception:  # pragma: no cover
    _HAS_GROUPED_MM = False


class BF16Experts:
    """Frozen experts kept as fused 3D tensors: gate_up [E, 2I, H], down [E, H, I]."""

    def __init__(self, gate_up: torch.Tensor, down: torch.Tensor):
        self.gate_up, self.down = gate_up, down
        self.num_experts = gate_up.shape[0]

    def weights(self, e0: int, e1: int):
        return self.gate_up[e0:e1], self.down[e0:e1]

    def tensors(self):
        return {"gate_up": self.gate_up, "down": self.down}

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors().values())


class Int4Experts:
    """Frozen experts in group-wise int4; dequantized on the fly per group of experts."""

    def __init__(self, gu: dict, dn: dict, dtype=torch.bfloat16):
        self.gu, self.dn, self.dtype = gu, dn, dtype
        self.num_experts = gu["q"].shape[0]

    @classmethod
    def from_bf16(cls, gate_up, down, group=64):
        return cls(quantize_int4(gate_up, group), quantize_int4(down, group), dtype=gate_up.dtype)

    def weights(self, e0: int, e1: int):
        gu = dequant_int4(self.gu["q"][e0:e1], self.gu["scale"][e0:e1], self.gu["zero"][e0:e1], self.dtype)
        dn = dequant_int4(self.dn["q"][e0:e1], self.dn["scale"][e0:e1], self.dn["zero"][e0:e1], self.dtype)
        return gu, dn

    def tensors(self):
        return {f"gate_up.{k}": v for k, v in self.gu.items()} | {f"down.{k}": v for k, v in self.dn.items()}

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors().values())


def _gmm(x: torch.Tensor, w: torch.Tensor, counts: list, transpose_w: bool = True):
    """Per-group matmul: rows of x are grouped contiguously by expert with sizes `counts`.
    Returns x_g @ w_g.T (transpose_w) or x_g @ w_g for each group, concatenated."""
    if x.shape[0] == 0:
        n = w.shape[1] if transpose_w else w.shape[2]
        return x.new_zeros(0, n)
    if _HAS_GROUPED_MM and x.dtype == torch.bfloat16 and x.is_cuda:
        offs = torch.tensor(counts, device=x.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        return torch._grouped_mm(x, w.transpose(1, 2) if transpose_w else w, offs=offs)
    outs, s = [], 0
    for g, c in enumerate(counts):
        if c:
            outs.append(x[s:s + c] @ (w[g].t() if transpose_w else w[g]))
            s += c
    return torch.cat(outs) if outs else x.new_zeros(0, w.shape[1] if transpose_w else w.shape[2])


def _groups(counts: list, group: int):
    """Yield (e0, e1, row_start, row_end, sub_counts) for consecutive expert groups that have tokens."""
    E = len(counts)
    row = 0
    for e0 in range(0, E, group):
        e1 = min(E, e0 + group)
        sub = counts[e0:e1]
        n = sum(sub)
        if n:
            yield e0, e1, row, row + n, sub
        row += n


_GELU_C = 0.7978845608028654  # sqrt(2 / pi)


def act(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "silu":
        return F.silu(x)
    if name == "gelu_tanh":
        return F.gelu(x, approximate="tanh")
    raise ValueError(f"unknown activation {name}")


def act_and_grad(g: torch.Tensor, name: str):
    """(act(g), d act / d g) in the dtype of g (called with fp32)."""
    if name == "silu":
        sg = torch.sigmoid(g)
        return g * sg, sg * (1 + g * (1 - sg))
    if name == "gelu_tanh":
        t = torch.tanh(_GELU_C * (g + 0.044715 * g ** 3))
        return 0.5 * g * (1 + t), 0.5 * (1 + t) + 0.5 * g * (1 - t * t) * _GELU_C * (1 + 3 * 0.044715 * g * g)
    raise ValueError(f"unknown activation {name}")


def frozen_experts_forward(xs, counts, bank, group, act_name="silu"):
    out = xs.new_empty(xs.shape[0], xs.shape[1])
    for e0, e1, r0, r1, sub in _groups(counts, group):
        gu, dn = bank.weights(e0, e1)
        h = _gmm(xs[r0:r1], gu, sub)
        g, u = h.chunk(2, dim=-1)
        out[r0:r1] = _gmm(act(g, act_name) * u, dn, sub)
    return out


class FrozenExpertsFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xs, counts, bank, group, act_name="silu"):
        ctx.save_for_backward(xs)
        ctx.counts, ctx.bank, ctx.group, ctx.act = counts, bank, group, act_name
        with torch.no_grad():
            return frozen_experts_forward(xs, counts, bank, group, act_name)

    @staticmethod
    def backward(ctx, dy):
        (xs,) = ctx.saved_tensors
        dx = torch.zeros_like(xs)
        dy = dy.to(xs.dtype)
        for e0, e1, r0, r1, sub in _groups(ctx.counts, ctx.group):
            gu, dn = ctx.bank.weights(e0, e1)
            h = _gmm(xs[r0:r1], gu, sub)
            g, u = h.float().chunk(2, dim=-1)
            da = _gmm(dy[r0:r1], dn, sub, transpose_w=False).float()  # y = a @ dn.T  ->  da = dy @ dn
            a, da_dg = act_and_grad(g, ctx.act)
            dg = da * u * da_dg
            du = da * a
            dh = torch.cat([dg, du], dim=-1).to(xs.dtype)
            dx[r0:r1] = _gmm(dh, gu, sub, transpose_w=False)
        return dx, None, None, None, None


class LayerExperts:
    """Routed experts of one layer: a frozen bank for all E experts plus optional trainable bf16 copies of the
    selected experts (their frozen-bank copy is then ignored)."""

    def __init__(self, frozen, sel_ids=None, sel_gate_up=None, sel_down=None, group: int = 32, act: str = "silu"):
        self.frozen = frozen
        self.E = frozen.num_experts
        self.group = group
        self.act = act
        self.sel_ids = None if sel_ids is None else torch.as_tensor(sel_ids, dtype=torch.long)
        self.sel_gate_up, self.sel_down = sel_gate_up, sel_down
        self._slot = {}
        if self.sel_ids is not None:
            self._slot = {int(e): j for j, e in enumerate(self.sel_ids.tolist())}


def route(x2d, gate_w, top_k):
    logits = F.linear(x2d, gate_w)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    topv, topi = torch.topk(probs, top_k, dim=-1)
    topv = topv / topv.sum(dim=-1, keepdim=True)
    return probs, topv.to(logits.dtype), topi


def route_gemma4(h, W, top_k, eps):
    """Gemma 4 router on the residual stream: RMSNorm without weight, x scale x H^-0.5 -> logits -> softmax (fp32)
    -> top-k, renormalized, x per_expert_scale of the chosen experts."""
    hf = h.float()
    x = (hf * torch.pow(hf.pow(2).mean(-1, keepdim=True) + eps, -0.5)).type_as(h)
    x = x * W["router.scale"] * h.shape[-1] ** -0.5
    logits = F.linear(x, W["router.proj.weight"])
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    topv, topi = torch.topk(probs, top_k, dim=-1)
    topv = topv / topv.sum(dim=-1, keepdim=True)
    return probs, topv * W["router.per_expert_scale"][topi], topi


def mlp_lora(x, w_gate, w_up, w_down, lora, act_name="silu"):
    """Gated MLP down(act(gate(x)) * up(x)) with optional LoRA on all three projections."""
    A = (lambda n: getattr(lora, "A_" + n)) if lora is not None else (lambda n: None)
    B = (lambda n: getattr(lora, "B_" + n)) if lora is not None else (lambda n: None)
    s = lora.scale if lora is not None else 0.0
    g = lora_linear(x, w_gate, A("gate"), B("gate"), s)
    u = lora_linear(x, w_up, A("up"), B("up"), s)
    return lora_linear(act(g, act_name) * u, w_down, A("down"), B("down"), s)


def shared_mlp(x, W, lora):
    pre = "mlp.shared_expert."
    y = mlp_lora(x, W[pre + "gate_proj.weight"], W[pre + "up_proj.weight"], W[pre + "down_proj.weight"], lora)
    return y * torch.sigmoid(F.linear(x, W["mlp.shared_expert_gate.weight"]))


def _selected_expert(x, gu, dn, lora, j, act_name="silu"):
    I = dn.shape[1]
    if lora is None:
        g, u = (x @ gu.t()).chunk(2, dim=-1)
        return (act(g, act_name) * u) @ dn.t()
    s = lora.scale
    g = lora_linear(x, gu[:I], lora.A_gate[j], lora.B_gate[j], s)
    u = lora_linear(x, gu[I:], lora.A_up[j], lora.B_up[j], s)
    return lora_linear(act(g, act_name) * u, dn, lora.A_down[j], lora.B_down[j], s)


def routed_experts(x2d, topv, topi, ex: LayerExperts, lora=None):
    """Weighted sum of routed experts for every token. Differentiable w.r.t. x2d, topv and LoRA params."""
    N, H = x2d.shape
    K = topi.shape[1]
    flat_e = topi.reshape(-1)
    order = torch.argsort(flat_e, stable=True)
    tok = order // K
    e_sorted = flat_e[order]
    counts_t = torch.bincount(flat_e, minlength=ex.E)
    xs = x2d[tok]

    if ex.sel_ids is None or len(ex._slot) == 0:
        y = FrozenExpertsFn.apply(xs, counts_t.tolist(), ex.frozen, ex.group, ex.act)
    else:
        sel_mask_e = torch.zeros(ex.E, dtype=torch.bool, device=x2d.device)
        sel_mask_e[ex.sel_ids.to(x2d.device)] = True
        row_sel = sel_mask_e[e_sorted]
        counts_f = counts_t.masked_fill(sel_mask_e, 0).tolist()
        rows, vals = [], []
        fr_rows = (~row_sel).nonzero().squeeze(1)
        if fr_rows.numel():
            rows.append(fr_rows)
            vals.append(FrozenExpertsFn.apply(xs[fr_rows], counts_f, ex.frozen, ex.group, ex.act))
        starts = (counts_t.cumsum(0) - counts_t).tolist()
        counts = counts_t.tolist()
        for e, j in ex._slot.items():
            c = counts[e]
            if c == 0:
                continue
            r0 = starts[e]
            rows.append(torch.arange(r0, r0 + c, device=x2d.device))
            vals.append(_selected_expert(xs[r0:r0 + c], ex.sel_gate_up[j], ex.sel_down[j], lora, j, ex.act)
                        .to(x2d.dtype))
        y = x2d.new_zeros(xs.shape[0], H).index_put((torch.cat(rows),), torch.cat(vals))
    # deterministic combine (no atomics): undo the sort and sum the K expert outputs of each token in fp32
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    y = y[inv].view(N, K, H)
    return (y.float() * topv.float().unsqueeze(-1)).sum(1).to(x2d.dtype)


def moe_forward(x2d, W, ex: LayerExperts, top_k: int, lora_layer=None, stats=None):
    """x2d [N, H] (already post-attention-normed). W holds router / shared expert tensors of the layer."""
    probs, topv, topi = route(x2d, W["mlp.gate.weight"], top_k)
    if stats is not None:
        stats(probs, topv, topi)
    lora_ex = lora_layer.experts if lora_layer is not None else None
    lora_sh = lora_layer.shared if lora_layer is not None else None
    routed = routed_experts(x2d, topv, topi, ex, lora_ex)
    return routed + shared_mlp(x2d, W, lora_sh)
