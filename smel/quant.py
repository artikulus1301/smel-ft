"""Group-wise asymmetric int4 quantization of frozen experts (two nibbles per byte, bf16 scale/zero) and int8 rows."""
import torch


def _q4(wg, mn, mx):
    scale = ((mx - mn) / 15).clamp_min(1e-8)
    # round scale/zero to bf16 first so quantization uses exactly the values stored
    scale = scale.to(torch.bfloat16).float()
    zero = mn.to(torch.bfloat16).float()
    q = torch.round((wg - zero) / scale).clamp_(0, 15)
    return q, scale, zero


def quantize_int4(w: torch.Tensor, group: int = 64, clip_search=(1.0, 0.95, 0.9, 0.85, 0.8)):
    """w [..., K] -> dict(q uint8 [..., K//2], scale/zero bf16 [..., K//group]); w ~= q * scale + zero.
    For every group the [min, max] range is shrunk by the factor from clip_search that minimizes the MSE.
    3D tensors (fused experts) are processed in slices of 16 experts to bound the fp32 temporaries."""
    if w.dim() == 3 and w.shape[0] > 16:
        parts = [quantize_int4(w[i:i + 16], group, clip_search) for i in range(0, w.shape[0], 16)]
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    K = w.shape[-1]
    assert K % group == 0 and group % 2 == 0
    wg = w.float().reshape(*w.shape[:-1], K // group, group)
    mn0 = wg.amin(-1, keepdim=True)
    mx0 = wg.amax(-1, keepdim=True)
    mid = (mn0 + mx0) / 2
    best = None
    for c in clip_search or (1.0,):
        mn, mx = mid + (mn0 - mid) * c, mid + (mx0 - mid) * c
        q, scale, zero = _q4(wg, mn, mx)
        err = (q * scale + zero - wg).pow(2).sum(-1, keepdim=True)
        if best is None:
            best = [err, q, scale, zero]
        else:
            better = err < best[0]
            best[0] = torch.where(better, err, best[0])
            best[1] = torch.where(better, q, best[1])
            best[2] = torch.where(better, scale, best[2])
            best[3] = torch.where(better, zero, best[3])
    _, q, scale, zero = best
    q = q.to(torch.uint8).reshape(*w.shape[:-1], K)
    packed = q[..., 0::2] | (q[..., 1::2] << 4)
    return {"q": packed.contiguous(), "scale": scale.squeeze(-1).to(torch.bfloat16),
            "zero": zero.squeeze(-1).to(torch.bfloat16)}


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _dq4_kernel(q_ptr, s_ptr, z_ptr, out_ptr, n_bytes, half_k, groups, GROUP: tl.constexpr,
                    BLOCK: tl.constexpr):
        # one program handles BLOCK packed bytes -> 2*BLOCK outputs
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n_bytes
        b = tl.load(q_ptr + offs, mask=m, other=0).to(tl.int32)
        row = offs // half_k
        col = (offs % half_k) * 2  # column of the low nibble
        gi = row * groups + col // GROUP  # both nibbles share a group (GROUP is even)
        s = tl.load(s_ptr + gi, mask=m, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + gi, mask=m, other=0.0).to(tl.float32)
        lo = (b & 0xF).to(tl.float32) * s + z
        hi = ((b >> 4) & 0xF).to(tl.float32) * s + z
        o = row * (half_k * 2) + col
        tl.store(out_ptr + o, lo.to(out_ptr.dtype.element_ty), mask=m)
        tl.store(out_ptr + o + 1, hi.to(out_ptr.dtype.element_ty), mask=m)

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


def dequant_int4(q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    if _HAS_TRITON and q.is_cuda and q.is_contiguous() and scale.is_contiguous() and zero.is_contiguous():
        half_k = q.shape[-1]
        groups = scale.shape[-1]
        out = torch.empty(*q.shape[:-1], half_k * 2, device=q.device, dtype=dtype)
        n = q.numel()
        BLOCK = 1024
        _dq4_kernel[(triton.cdiv(n, BLOCK),)](q, scale, zero, out, n, half_k, groups,
                                              GROUP=(half_k * 2) // groups, BLOCK=BLOCK)
        return out
    return dequant_int4_torch(q, scale, zero, dtype)


def dequant_int4_torch(q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, dtype=torch.bfloat16):
    K = q.shape[-1] * 2
    G = scale.shape[-1]
    lo = (q & 0xF)
    hi = (q >> 4)
    w = torch.stack((lo, hi), dim=-1).reshape(*q.shape[:-1], G, K // G).to(dtype)
    w = w * scale.unsqueeze(-1).to(dtype) + zero.unsqueeze(-1).to(dtype)
    return w.reshape(*q.shape[:-1], K)


def quantize_int8_rows(w: torch.Tensor):
    """Symmetric per-row int8 (for lm_head / dense matrices)."""
    s = (w.float().abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
    q = torch.round(w.float() / s).clamp_(-127, 127).to(torch.int8)
    return {"q": q, "scale": s.squeeze(-1).to(torch.bfloat16)}


def dequant_int8_rows(q, scale, dtype=torch.bfloat16):
    return q.to(dtype) * scale.unsqueeze(-1).to(dtype)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()
