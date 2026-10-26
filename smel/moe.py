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


def frozen_experts_forward(xs, counts, bank, group):
    out = xs.new_empty(xs.shape[0], xs.shape[1])
    for e0, e1, r0, r1, sub in _groups(counts, group):
        gu, dn = bank.weights(e0, e1)
        h = _gmm(xs[r0:r1], gu, sub)
        g, u = h.chunk(2, dim=-1)
        out[r0:r1] = _gmm(F.silu(g) * u, dn, sub)
    return out


class FrozenExpertsFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xs, counts, bank, group):
        ctx.save_for_backward(xs)
        ctx.counts, ctx.bank, ctx.group = counts, bank, group
        with torch.no_grad():
            return frozen_experts_forward(xs, counts, bank, group)

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
            sg = torch.sigmoid(g)
            silu = g * sg
            dg = da * u * (sg * (1 + g * (1 - sg)))
            du = da * silu
            dh = torch.cat([dg, du], dim=-1).to(xs.dtype)
            dx[r0:r1] = _gmm(dh, gu, sub, transpose_w=False)
        return dx, None, None, None


class LayerExperts:
    """Routed experts of one layer: a frozen bank for all E experts plus optional trainable bf16 copies of the
    selected experts (their frozen-bank copy is then ignored)."""

    def __init__(self, frozen, sel_ids=None, sel_gate_up=None, sel_down=None, group: int = 32):
        self.frozen = frozen
        self.E = frozen.num_experts
        self.group = group
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


def shared_mlp(x, W, lora):
    pre = "mlp.shared_expert."
    A = (lambda n: getattr(lora, "A_" + n)) if lora is not None else (lambda n: None)
    B = (lambda n: getattr(lora, "B_" + n)) if lora is not None else (lambda n: None)
    s = lora.scale if lora is not None else 0.0
    g = lora_linear(x, W[pre + "gate_proj.weight"], A("gate"), B("gate"), s)
    u = lora_linear(x, W[pre + "up_proj.weight"], A("up"), B("up"), s)
    y = lora_linear(F.silu(g) * u, W[pre + "down_proj.weight"], A("down"), B("down"), s)
    return y * torch.sigmoid(F.linear(x, W["mlp.shared_expert_gate.weight"]))


def _selected_expert(x, gu, dn, lora, j):
    I = dn.shape[1]
    if lora is None:
        g, u = (x @ gu.t()).chunk(2, dim=-1)
        return (F.silu(g) * u) @ dn.t()
    s = lora.scale
    g = lora_linear(x, gu[:I], lora.A_gate[j], lora.B_gate[j], s)
    u = lora_linear(x, gu[I:], lora.A_up[j], lora.B_up[j], s)
    return lora_linear(F.silu(g) * u, dn, lora.A_down[j], lora.B_down[j], s)


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
        y = FrozenExpertsFn.apply(xs, counts_t.tolist(), ex.frozen, ex.group)
    else:
        sel_mask_e = torch.zeros(ex.E, dtype=torch.bool, device=x2d.device)
        sel_mask_e[ex.sel_ids.to(x2d.device)] = True
        row_sel = sel_mask_e[e_sorted]
        counts_f = counts_t.masked_fill(sel_mask_e, 0).tolist()
        rows, vals = [], []
        fr_rows = (~row_sel).nonzero().squeeze(1)
        if fr_rows.numel():
            rows.append(fr_rows)
            vals.append(FrozenExpertsFn.apply(xs[fr_rows], counts_f, ex.frozen, ex.group))
        starts = (counts_t.cumsum(0) - counts_t).tolist()
        counts = counts_t.tolist()
        for e, j in ex._slot.items():
            c = counts[e]
            if c == 0:
                continue
            r0 = starts[e]
            rows.append(torch.arange(r0, r0 + c, device=x2d.device))
            vals.append(_selected_expert(xs[r0:r0 + c], ex.sel_gate_up[j], ex.sel_down[j], lora, j).to(x2d.dtype))
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
