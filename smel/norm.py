"""RMSNorm of the frozen layers as Triton kernels: one launch forward, one backward, instead of the 5-15 elementwise
and reduction kernels of the torch code (each a full pass over the activations in fp32).

  rmsnorm(x, w, eps, plus_one)       y = x * rstd * (1 + w)  (Qwen3.5/3.6), x * rstd * w (Gemma 4), or x * rstd (w None)
  add_rmsnorm(h, x, w, eps)          s = h + x (the residual add), y = rmsnorm(s): one kernel each way
  gated_rmsnorm(x, z, w, eps)        y = w * norm(x) * silu(z)  (output norm of Gated DeltaNet)

The norm weights are frozen in SMEL (LoRA trains experts / MLPs only), so backward gives dx (and dz) only. Every
rounding of the torch code is kept (fp32 math, bf16 where the torch code rounds), so the result equals the torch
path up to the last bit of fp32 rsqrt; tests/test_norm.py compares both. Saved for backward: the input itself
(already alive as the residual stream) and one fp32 rstd per row.

    xhat = x * rstd,  g = dy * w',  dx = rstd * (g - xhat * mean(g * xhat))     (w' = 1 + w, w or 1)

The row count is not specialized on, so microbatches of different lengths reuse one compiled kernel.
Ideas: the RMSNorm formula is Zhang & Sennrich, 2019 (https://arxiv.org/abs/1910.07467); one-pass fused norm kernels
are common (Triton's layer-norm tutorial, Liger-Kernel). The fused residual add follows the author's kernels written
for a fine-tuning hackathon (norm.py there); code written here for the bf16 residual stream and the three variants.
"""
import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False

MAX_COLS = 16384


def _tile(n_cols):
    block = triton.next_power_of_2(n_cols)
    rows = max(1, min(64, 4096 // block))          # small rows (head dims): several rows per program
    warps = 8 if block * rows >= 8192 else 4
    return block, rows, warps


if _HAS_TRITON:
    @triton.jit
    def _w_eff(w_ptr, cols, inside, W_MODE: tl.constexpr):
        # W_MODE 0: no weight, 1: w, 2: 1 + w
        if W_MODE == 0:
            return tl.full(cols.shape, 1.0, tl.float32)
        w = tl.load(w_ptr + cols, mask=inside, other=0.0).to(tl.float32)
        if W_MODE == 2:
            w = 1.0 + w
        return w

    @triton.jit(do_not_specialize=["n_rows"])
    def _rms_fwd(x_ptr, w_ptr, y_ptr, rstd_ptr, n_rows, n_cols, x_stride, eps,
                 W_MODE: tl.constexpr, BLOCK: tl.constexpr, ROWS: tl.constexpr):
        rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK)
        inside = cols < n_cols
        ok = (rows < n_rows)[:, None] & inside[None, :]
        x = tl.load(x_ptr + rows[:, None] * x_stride + cols[None, :], mask=ok, other=0.0).to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / n_cols + eps)
        w = _w_eff(w_ptr, cols, inside, W_MODE)
        y = x * rstd[:, None] * w[None, :]
        tl.store(y_ptr + rows[:, None] * n_cols + cols[None, :], y.to(y_ptr.dtype.element_ty), mask=ok)
        tl.store(rstd_ptr + rows, rstd, mask=rows < n_rows)

    @triton.jit(do_not_specialize=["n_rows"])
    def _rms_bwd(dy_ptr, x_ptr, w_ptr, rstd_ptr, ds_ptr, dx_ptr, n_rows, n_cols, x_stride,
                 W_MODE: tl.constexpr, HAS_DS: tl.constexpr, BLOCK: tl.constexpr, ROWS: tl.constexpr):
        rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK)
        inside = cols < n_cols
        ok = (rows < n_rows)[:, None] & inside[None, :]
        off = rows[:, None] * n_cols + cols[None, :]
        dy = tl.load(dy_ptr + off, mask=ok, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + rows[:, None] * x_stride + cols[None, :], mask=ok, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows, mask=rows < n_rows, other=0.0)
        w = _w_eff(w_ptr, cols, inside, W_MODE)
        xhat = x * rstd[:, None]
        g = dy * w[None, :]
        c = tl.sum(g * xhat, axis=1) / n_cols
        dx = rstd[:, None] * (g - xhat * c[:, None])
        if HAS_DS:  # gradient that reaches the residual sum from its other use
            dx += tl.load(ds_ptr + off, mask=ok, other=0.0).to(tl.float32)
        tl.store(dx_ptr + off, dx.to(dx_ptr.dtype.element_ty), mask=ok)

    @triton.jit(do_not_specialize=["n_rows"])
    def _add_rms_fwd(h_ptr, x_ptr, w_ptr, s_ptr, y_ptr, rstd_ptr, n_rows, n_cols, eps,
                     W_MODE: tl.constexpr, BLOCK: tl.constexpr, ROWS: tl.constexpr):
        rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK)
        inside = cols < n_cols
        ok = (rows < n_rows)[:, None] & inside[None, :]
        off = rows[:, None] * n_cols + cols[None, :]
        h = tl.load(h_ptr + off, mask=ok, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + off, mask=ok, other=0.0).to(tl.float32)
        s = (h + x).to(s_ptr.dtype.element_ty)           # the residual stream is rounded like torch's h + x
        tl.store(s_ptr + off, s, mask=ok)
        s = s.to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(s * s, axis=1) / n_cols + eps)
        w = _w_eff(w_ptr, cols, inside, W_MODE)
        tl.store(y_ptr + off, (s * rstd[:, None] * w[None, :]).to(y_ptr.dtype.element_ty), mask=ok)
        tl.store(rstd_ptr + rows, rstd, mask=rows < n_rows)

    @triton.jit(do_not_specialize=["n_rows"])
    def _gated_fwd(x_ptr, z_ptr, w_ptr, y_ptr, rstd_ptr, n_rows, n_cols, eps,
                   BLOCK: tl.constexpr, ROWS: tl.constexpr):
        rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK)
        inside = cols < n_cols
        ok = (rows < n_rows)[:, None] & inside[None, :]
        off = rows[:, None] * n_cols + cols[None, :]
        x = tl.load(x_ptr + off, mask=ok, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + off, mask=ok, other=0.0).to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / n_cols + eps)
        w = tl.load(w_ptr + cols, mask=inside, other=0.0).to(tl.float32)
        # torch: n = (x * rstd).to(dt); y = w * n (rounded to dt); out = (y.float() * silu(z.float())).to(dt)
        n = (x * rstd[:, None]).to(y_ptr.dtype.element_ty).to(tl.float32)
        y = (w[None, :] * n).to(y_ptr.dtype.element_ty).to(tl.float32)
        out = y * (z * tl.sigmoid(z))
        tl.store(y_ptr + off, out.to(y_ptr.dtype.element_ty), mask=ok)
        tl.store(rstd_ptr + rows, rstd, mask=rows < n_rows)

    @triton.jit(do_not_specialize=["n_rows"])
    def _gated_bwd(do_ptr, x_ptr, z_ptr, w_ptr, rstd_ptr, dx_ptr, dz_ptr, n_rows, n_cols,
                   BLOCK: tl.constexpr, ROWS: tl.constexpr):
        rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK)
        inside = cols < n_cols
        ok = (rows < n_rows)[:, None] & inside[None, :]
        off = rows[:, None] * n_cols + cols[None, :]
        do = tl.load(do_ptr + off, mask=ok, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + off, mask=ok, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + off, mask=ok, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows, mask=rows < n_rows, other=0.0)
        w = tl.load(w_ptr + cols, mask=inside, other=0.0).to(tl.float32)
        xhat = x * rstd[:, None]
        n = xhat.to(dx_ptr.dtype.element_ty).to(tl.float32)
        y = (w[None, :] * n).to(dx_ptr.dtype.element_ty).to(tl.float32)
        sg = tl.sigmoid(z)
        silu = z * sg
        dz = do * y * (sg * (1.0 + z * (1.0 - sg)))
        dy = (do * silu).to(dx_ptr.dtype.element_ty).to(tl.float32)         # grad of the bf16 y
        dn = (dy * w[None, :]).to(dx_ptr.dtype.element_ty).to(tl.float32)   # grad of the bf16 n
        c = tl.sum(dn * xhat, axis=1) / n_cols
        dx = rstd[:, None] * (dn - xhat * c[:, None])
        tl.store(dx_ptr + off, dx.to(dx_ptr.dtype.element_ty), mask=ok)
        tl.store(dz_ptr + off, dz.to(dz_ptr.dtype.element_ty), mask=ok)


def _rows(x):
    """x as [rows, n_cols] with unit column stride, without a copy when possible; (x2d, row stride)."""
    n_cols = x.shape[-1]
    x2 = x.reshape(-1, n_cols)
    if x2.stride(-1) != 1 or (x2.shape[0] > 1 and x2.stride(0) < n_cols):
        x2 = x2.contiguous()
    return x2, (x2.stride(0) if x2.shape[0] > 1 else n_cols)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps, w_mode):
        x2, xs = _rows(x)
        rows, n_cols = x2.shape
        block, per, warps = _tile(n_cols)
        y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        wt = w if w is not None else rstd
        if rows:
            _rms_fwd[(triton.cdiv(rows, per),)](x2, wt, y, rstd, rows, n_cols, xs, float(eps), W_MODE=w_mode,
                                                BLOCK=block, ROWS=per, num_warps=warps)
        ctx.save_for_backward(x2, wt, rstd)
        ctx.shape, ctx.xs, ctx.w_mode = x.shape, xs, w_mode
        return y

    @staticmethod
    def backward(ctx, dy):
        x2, wt, rstd = ctx.saved_tensors
        rows, n_cols = x2.shape
        block, per, warps = _tile(n_cols)
        dy = dy.reshape(rows, n_cols).contiguous()
        dx = torch.empty(ctx.shape, device=dy.device, dtype=x2.dtype)
        if rows:
            _rms_bwd[(triton.cdiv(rows, per),)](dy, x2, wt, rstd, dy, dx, rows, n_cols, ctx.xs, W_MODE=ctx.w_mode,
                                                HAS_DS=False, BLOCK=block, ROWS=per, num_warps=warps)
        return dx, None, None, None


class _AddRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, x, w, eps, w_mode):
        n_cols = h.shape[-1]
        h2, x2 = h.reshape(-1, n_cols).contiguous(), x.reshape(-1, n_cols).contiguous()
        rows = h2.shape[0]
        block, per, warps = _tile(n_cols)
        s = torch.empty(h.shape, device=h.device, dtype=h.dtype)
        y = torch.empty(h.shape, device=h.device, dtype=h.dtype)
        rstd = torch.empty(rows, device=h.device, dtype=torch.float32)
        if rows:
            _add_rms_fwd[(triton.cdiv(rows, per),)](h2, x2, w, s, y, rstd, rows, n_cols, float(eps), W_MODE=w_mode,
                                                    BLOCK=block, ROWS=per, num_warps=warps)
        ctx.save_for_backward(s, w, rstd)
        ctx.w_mode = w_mode
        ctx.set_materialize_grads(False)
        return s, y

    @staticmethod
    def backward(ctx, ds, dy):
        s, w, rstd = ctx.saved_tensors
        if dy is None:
            return ds, ds, None, None, None
        n_cols = s.shape[-1]
        s2 = s.reshape(-1, n_cols)
        rows = s2.shape[0]
        block, per, warps = _tile(n_cols)
        dy = dy.reshape(rows, n_cols).contiguous()
        ds2 = ds.reshape(rows, n_cols).contiguous() if ds is not None else dy
        d = torch.empty(s.shape, device=s.device, dtype=s.dtype)
        if rows:
            _rms_bwd[(triton.cdiv(rows, per),)](dy, s2, w, rstd, ds2, d, rows, n_cols, n_cols, W_MODE=ctx.w_mode,
                                                HAS_DS=ds is not None, BLOCK=block, ROWS=per, num_warps=warps)
        return d, d, None, None, None


class _GatedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, z, w, eps):
        n_cols = x.shape[-1]
        x2, z2 = x.reshape(-1, n_cols).contiguous(), z.reshape(-1, n_cols).contiguous()
        rows = x2.shape[0]
        block, per, warps = _tile(n_cols)
        y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        if rows:
            _gated_fwd[(triton.cdiv(rows, per),)](x2, z2, w, y, rstd, rows, n_cols, float(eps), BLOCK=block,
                                                  ROWS=per, num_warps=warps)
        ctx.save_for_backward(x2, z2, w, rstd)
        ctx.shapes = (x.shape, z.shape)
        return y

    @staticmethod
    def backward(ctx, do):
        x2, z2, w, rstd = ctx.saved_tensors
        rows, n_cols = x2.shape
        block, per, warps = _tile(n_cols)
        do = do.reshape(rows, n_cols).contiguous()
        dx = torch.empty(ctx.shapes[0], device=do.device, dtype=x2.dtype)
        dz = torch.empty(ctx.shapes[1], device=do.device, dtype=z2.dtype)
        if rows:
            _gated_bwd[(triton.cdiv(rows, per),)](do, x2, z2, w, rstd, dx, dz, rows, n_cols, BLOCK=block, ROWS=per,
                                                  num_warps=warps)
        return dx, dz, None, None


def supported(x, *weights):
    """Kernels apply to a CUDA bf16/fp16 input whose norm weights are frozen and of the input's dtype."""
    if not (_HAS_TRITON and ENABLED and x.is_cuda and x.dtype in (torch.bfloat16, torch.float16)):
        return False
    if x.shape[-1] > MAX_COLS:
        return False
    return all(w is None or (w.dtype == x.dtype and not w.requires_grad and w.is_contiguous()
                             and w.shape == (x.shape[-1],)) for w in weights)


ENABLED = True   # tests switch the kernels off to compare with the torch code


def rmsnorm(x, w, eps, plus_one):
    return _RMSNorm.apply(x, w, eps, 0 if w is None else (2 if plus_one else 1))


def add_rmsnorm(h, x, w, eps, plus_one):
    """(h + x, rmsnorm(h + x)): the residual add and the norm after it."""
    return _AddRMSNorm.apply(h, x, w, eps, 2 if plus_one else 1)


def gated_rmsnorm(x, z, w, eps):
    return _GatedRMSNorm.apply(x, z, w, eps)
