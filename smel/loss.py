"""Chunked cross-entropy over a frozen lm_head (bf16 or int8 rows) without materializing [T, 248k] logits.

The gradient w.r.t. the hidden states is computed in the same pass as the loss (lm_head is frozen) and returned by
backward, so peak memory is one [token_chunk, vocab_chunk] fp32 tile. Optional final-logit softcapping
(Gemma: logits = cap * tanh(z / cap)) is applied inside the same pass, gradient included.
"""
import torch

from .quant import dequant_int8_rows, quantize_int8_rows


class LMHead:
    def __init__(self, weight: torch.Tensor | None = None, int8: dict | None = None, vocab_chunk: int = 32768,
                 softcap: float = 0.0):
        assert (weight is None) != (int8 is None)
        self.w, self.q = weight, int8
        self.vocab = weight.shape[0] if weight is not None else int8["q"].shape[0]
        self.vocab_chunk = vocab_chunk
        self.softcap = float(softcap or 0.0)

    @classmethod
    def from_weight(cls, w, int8=False, vocab_chunk=32768, softcap=0.0):
        if int8:
            return cls(int8=quantize_int8_rows(w), vocab_chunk=vocab_chunk, softcap=softcap)
        return cls(weight=w, vocab_chunk=vocab_chunk, softcap=softcap)

    def to(self, device):
        if self.w is not None:
            return LMHead(weight=self.w.to(device), vocab_chunk=self.vocab_chunk, softcap=self.softcap)
        return LMHead(int8={k: v.to(device) for k, v in self.q.items()}, vocab_chunk=self.vocab_chunk,
                      softcap=self.softcap)

    def cap(self, z: torch.Tensor) -> torch.Tensor:
        return self.softcap * torch.tanh(z / self.softcap) if self.softcap else z

    def rows(self, v0, v1, dtype):
        if self.w is not None:
            return self.w[v0:v1].to(dtype)
        return dequant_int8_rows(self.q["q"][v0:v1], self.q["scale"][v0:v1], dtype)

    def chunks(self):
        for v0 in range(0, self.vocab, self.vocab_chunk):
            yield v0, min(self.vocab, v0 + self.vocab_chunk)

    def logits(self, h):
        return torch.cat([self.cap((h @ self.rows(v0, v1, h.dtype).t()).float()) for v0, v1 in self.chunks()],
                         dim=-1)


def _ce_chunk(h, head: LMHead, tgt, need_grad):
    """Loss sum and (optionally) d loss_sum / d h for one token chunk, in ONE pass over the vocabulary.

    Idea (Soup Kernel Challenge, fused_ce.py: the gradient is taken in the forward pass, written for SMEL FT): the
    gradient sum_v softmax_v * W_v is accumulated online with the same running maximum as the log-sum-exp: when the
    maximum moves from m to m', the accumulator and the running sum are both multiplied by exp(m - m'). The target
    row W[tgt] (and, with softcapping, the derivative of the cap there) is taken from the chunk that holds the
    target. The two-pass version (_ce_chunk_two_pass) reads the head weight twice; this one reads it once."""
    n, H = h.shape
    m = torch.full((n,), -float("inf"), device=h.device)
    s = torch.zeros(n, device=h.device)
    tgt_logit = torch.zeros(n, device=h.device)
    acc = torch.zeros(n, H, device=h.device) if need_grad else None
    tgt_row = torch.zeros(n, H, device=h.device) if need_grad else None
    tgt_d = torch.ones(n, device=h.device) if need_grad else None   # d(cap)/dz at the target (1 without cap)
    for v0, v1 in head.chunks():
        W = head.rows(v0, v1, h.dtype)
        lg = head.cap((h @ W.t()).float())
        cm = lg.amax(-1)
        nm = torch.maximum(m, cm)
        r = (m - nm).exp()                       # rescale of the running sums to the new maximum
        p = (lg - nm[:, None]).exp()
        s = s * r + p.sum(-1)
        inside = (tgt >= v0) & (tgt < v1)
        if inside.any():
            idx = inside.nonzero().squeeze(1)
            col = tgt[idx] - v0
            tgt_logit[idx] = lg[idx, col]
            if need_grad:
                tgt_row[idx] = W[col].float()
                if head.softcap:
                    tgt_d[idx] = 1 - (lg[idx, col] / head.softcap) ** 2
        if need_grad:
            if head.softcap:
                p = p * (1 - (lg / head.softcap) ** 2)   # d cap(z) / dz, applied to every softmax term
            acc = acc * r[:, None] + (p.to(h.dtype) @ W).float()
        m = nm
    lse = m + s.log()
    loss = (lse - tgt_logit).sum()
    if not need_grad:
        return loss, None
    gh = acc / s[:, None] - tgt_d[:, None] * tgt_row   # sum_v (softmax_v - onehot_v) * dcap_v * W_v
    return loss, gh


def _ce_chunk_two_pass(h, head: LMHead, tgt, need_grad):
    """Loss sum and (optionally) d loss_sum / d h for one token chunk; two passes over vocab chunks (reference)."""
    n = h.shape[0]
    m = torch.full((n,), -float("inf"), device=h.device)
    s = torch.zeros(n, device=h.device)
    tgt_logit = torch.zeros(n, device=h.device)
    for v0, v1 in head.chunks():
        lg = head.cap((h @ head.rows(v0, v1, h.dtype).t()).float())
        cm = lg.amax(-1)
        nm = torch.maximum(m, cm)
        s = s * (m - nm).exp() + (lg - nm[:, None]).exp().sum(-1)
        m = nm
        inside = (tgt >= v0) & (tgt < v1)
        if inside.any():
            idx = inside.nonzero().squeeze(1)
            tgt_logit[idx] = lg[idx, tgt[idx] - v0]
    lse = m + s.log()
    loss = (lse - tgt_logit).sum()
    if not need_grad:
        return loss, None
    gh = torch.zeros(h.shape, device=h.device, dtype=torch.float32)
    for v0, v1 in head.chunks():
        W = head.rows(v0, v1, h.dtype)
        lg = head.cap((h @ W.t()).float())
        p = (lg - lse[:, None]).exp()
        inside = (tgt >= v0) & (tgt < v1)
        if inside.any():
            idx = inside.nonzero().squeeze(1)
            p[idx, tgt[idx] - v0] -= 1.0
        if head.softcap:
            p = p * (1 - (lg / head.softcap) ** 2)   # d cap*tanh(z/cap) / dz
        gh += p.to(h.dtype) @ W
    return loss, gh


class _ChunkedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, head, tgt, denom, token_chunk, need_grad):
        total = torch.zeros((), device=h.device)
        grads = [] if need_grad else None
        for t0 in range(0, h.shape[0], token_chunk):
            l, g = _ce_chunk(h[t0:t0 + token_chunk], head, tgt[t0:t0 + token_chunk], need_grad)
            total += l
            if need_grad:
                grads.append(g)
        if need_grad:
            ctx.grad = (torch.cat(grads) / denom).to(h.dtype) if grads else torch.zeros_like(h)
        return total / denom

    @staticmethod
    def backward(ctx, go):
        return ctx.grad * go.to(ctx.grad.dtype), None, None, None, None, None


def chunked_ce(h, head: LMHead, tgt, denom: float, token_chunk: int = 1024, need_grad: bool = True):
    """sum_t CE(h_t @ W.T, tgt_t) / denom. h: [N, H] (final-normed, only supervised rows), tgt: [N] long."""
    if h.shape[0] == 0:
        return (h.sum() * 0.0)
    if need_grad and h.requires_grad:
        return _ChunkedCE.apply(h, head, tgt, float(denom), token_chunk, True)
    with torch.no_grad():
        return _ChunkedCE.apply(h, head, tgt, float(denom), token_chunk, False)
