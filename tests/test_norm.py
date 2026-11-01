"""Triton norms (smel/norm.py) against the torch code of layer_runner: forward and gradients of rmsnorm (1 + w, w,
no weight), the fused residual add + norm, and the gated output norm of Gated DeltaNet; plus the routing in
layer_runner (kernels for bf16 on CUDA, torch for fp32 / trainable weights)."""
import pytest
import torch

from smel import layer_runner as R
from smel import norm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need CUDA")
BF, EPS = torch.bfloat16, 1e-6


def torch_path(fn, *a, **kw):
    """Run a layer_runner function with the kernels switched off."""
    norm.ENABLED = False
    try:
        return fn(*a, **kw)
    finally:
        norm.ENABLED = True


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def grads(fn, inputs, cot):
    xs = [t.detach().clone().requires_grad_(True) for t in inputs]
    out = fn(*xs)
    out = out if isinstance(out, tuple) else (out,)
    sum((o.float() * c.float()).sum() for o, c in zip(out, cot)).backward()
    return out, [x.grad for x in xs]


@pytest.mark.parametrize("shape", [(37, 2048), (3, 29, 2816), (64, 16, 256), (50, 2, 256), (40, 100)])
@pytest.mark.parametrize("variant", ["plus_one", "scale", "none"])
def test_rmsnorm_matches_torch(shape, variant):
    torch.manual_seed(0)
    x = (torch.randn(shape, device="cuda") * 2).to(BF)
    w = None if variant == "none" else (torch.randn(shape[-1], device="cuda") * 0.1).to(BF)
    plus_one = variant == "plus_one"
    assert norm.supported(x, w)
    cot = [torch.randn(shape, device="cuda").to(BF)]
    (y,), (dx,) = grads(lambda t: R.rmsnorm(t, w, EPS, plus_one), [x], cot)
    (yr,), (dxr,) = torch_path(grads, lambda t: R.rmsnorm(t, w, EPS, plus_one), [x], cot)
    assert y.dtype == BF and y.shape == x.shape
    assert rel(y, yr) < 1e-4 and rel(dx, dxr) < 1e-4


def test_rmsnorm_of_a_strided_view():
    """q of the gated attention is a chunk of q_proj's output: rows with a stride of 2 * head_dim."""
    torch.manual_seed(1)
    qg = torch.randn(70, 16, 512, device="cuda").to(BF)
    w = (torch.randn(256, device="cuda") * 0.1).to(BF)
    cot = [torch.randn(70, 16, 256, device="cuda").to(BF)]
    f = lambda t: R.rmsnorm(t.chunk(2, dim=-1)[0], w, EPS)
    (y,), (dg,) = grads(f, [qg], cot)
    (yr,), (dgr,) = torch_path(grads, f, [qg], cot)
    assert rel(y, yr) < 1e-4 and rel(dg, dgr) < 1e-4
    assert dg[..., 256:].abs().max() == 0


@pytest.mark.parametrize("plus_one", [True, False])
def test_add_rmsnorm_matches_fp64(plus_one):
    """s = h + x is rounded to bf16 exactly like torch; y and the gradients agree with an fp64 reference at least as
    well as the torch bf16 code does (the kernel adds the two gradients of s in fp32, torch in bf16)."""
    torch.manual_seed(2)
    h = torch.randn(300, 2048, device="cuda").to(BF)
    x = torch.randn_like(h)
    w = (torch.randn(2048, device="cuda") * 0.1).to(BF)
    cot = [torch.randn_like(h), torch.randn_like(h)]
    f = lambda a, b: R.add_rmsnorm(a, b, w, EPS, plus_one)
    (s, y), (dh, dx) = grads(f, [h, x], cot)
    (sr, yr), (dhr, dxr) = torch_path(grads, f, [h, x], cot)
    assert torch.equal(s, sr)
    assert torch.equal(dh, dx)

    def ref64(a, b):
        s64 = (a + b).to(BF).double()     # same bf16 residual as both implementations
        n = s64 * torch.rsqrt(s64.pow(2).mean(-1, keepdim=True) + EPS)
        return s64, n * ((1 + w.double()) if plus_one else w.double())
    hd = h.double().requires_grad_(True)
    s64, y64 = ref64(hd, x.double())
    ((s64 * cot[0].double()).sum() + (y64 * cot[1].double()).sum()).backward()
    assert rel(y, y64) < 5e-3
    assert rel(dh, hd.grad) <= rel(dhr, hd.grad) * 1.05 + 1e-6


def test_gated_rmsnorm_matches_torch():
    torch.manual_seed(3)
    x = torch.randn(64 * 32, 128, device="cuda").to(BF)
    z = torch.randn_like(x)
    w = (1 + 0.1 * torch.randn(128, device="cuda")).to(BF)
    cot = [torch.randn_like(x)]
    f = lambda a, b: R.gated_rmsnorm(a, b, w, EPS)
    (y,), (dx, dz) = grads(f, [x, z], cot)
    (yr,), (dxr, dzr) = torch_path(grads, f, [x, z], cot)
    assert rel(y, yr) < 1e-3 and rel(dx, dxr) < 1e-3 and rel(dz, dzr) < 1e-3


def test_fallbacks():
    x32 = torch.randn(8, 64, device="cuda")
    assert not norm.supported(x32, torch.ones(64, device="cuda"))           # fp32: exact torch path
    xb = x32.to(BF)
    assert not norm.supported(xb, torch.ones(64, device="cuda", dtype=BF, requires_grad=True))
    assert not norm.supported(xb, torch.ones(64, device="cuda"))            # weight of another dtype
    assert not norm.supported(xb.cpu(), None)
    wt = torch.ones(64, device="cuda", dtype=BF, requires_grad=True)        # trainable weight still trains (torch)
    R.rmsnorm(xb, wt, EPS).float().sum().backward()
    assert wt.grad is not None and torch.isfinite(wt.grad).all()
