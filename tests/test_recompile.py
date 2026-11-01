"""No Triton kernel of SMEL compiles again for a new microbatch length (the packed token count changes every step).
Idea: the check_recompile tool of the authors' Soup Kernel Challenge solution; here a test that counts the compiled
variants in each kernel's JIT cache after the first length and after all of them."""
import pytest
import torch

from smel import norm, quant
from smel.layer_runner import add_rmsnorm, gated_rmsnorm, rmsnorm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need CUDA")
BF = torch.bfloat16
LENGTHS = [37, 1, 140, 512, 1500, 4092]


def variants(kernel):
    return sum(len(c[0]) for c in kernel.device_caches.values())


def step(T):
    """Forward + backward of every norm kernel at the shapes of Qwen3.6 / Gemma 4 for T tokens."""
    dev = "cuda"
    for shape, plus_one, has_w in (((T, 2048), True, True), ((T, 16, 256), True, True), ((T, 2816), False, True),
                                   ((T, 8, 256), False, False)):
        x = torch.randn(shape, device=dev, dtype=BF, requires_grad=True)
        w = torch.randn(shape[-1], device=dev, dtype=BF) if has_w else None
        rmsnorm(x, w, 1e-6, plus_one).float().sum().backward()
    h = torch.randn(T, 2048, device=dev, dtype=BF, requires_grad=True)
    a = torch.randn(T, 2048, device=dev, dtype=BF, requires_grad=True)
    s, y = add_rmsnorm(h, a, torch.randn(2048, device=dev, dtype=BF), 1e-6)
    (s.float().sum() + y.float().sum()).backward()
    x = torch.randn(T * 32, 128, device=dev, dtype=BF, requires_grad=True)
    z = torch.randn(T * 32, 128, device=dev, dtype=BF, requires_grad=True)
    gated_rmsnorm(x, z, torch.randn(128, device=dev, dtype=BF), 1e-6).float().sum().backward()


def test_norm_kernels_do_not_recompile_per_length():
    kernels = [norm._rms_fwd, norm._rms_bwd, norm._add_rms_fwd, norm._gated_fwd, norm._gated_bwd]
    step(LENGTHS[0])
    torch.cuda.synchronize()
    first = [variants(k) for k in kernels]
    assert all(n > 0 for n in first)
    for T in LENGTHS[1:]:
        step(T)
    torch.cuda.synchronize()
    assert [variants(k) for k in kernels] == first


def test_int4_dequant_does_not_recompile_per_expert_count():
    """Int4Experts.weights dequantizes groups of experts; the last group of a layer is smaller."""
    w = torch.randn(9, 64, 128, device="cuda", dtype=BF)
    Q = quant.quantize_int4(w, 64)
    quant.dequant_int4(Q["q"][:8], Q["scale"][:8], Q["zero"][:8])
    first = variants(quant._dq4_kernel)
    for e1 in (1, 3, 5, 9):
        quant.dequant_int4(Q["q"][:e1], Q["scale"][:e1], Q["zero"][:e1])
    torch.cuda.synchronize()
    assert variants(quant._dq4_kernel) == first
