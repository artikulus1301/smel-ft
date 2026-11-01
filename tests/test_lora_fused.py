"""LoRA GEMM fusions (smel/lora.py): addmm epilogue and the shared gate/up input against the plain formula, and the
stacked gate/up weight of LayerWeights."""
import torch

from smel.layer_runner import LayerWeights
from smel.lora import lora_gate_up, lora_linear


def plain(x, w, A, B, s):
    return x @ w.t() + s * ((x @ A.t()) @ B.t())


def test_lora_linear_and_gate_up_match_plain_formula():
    torch.manual_seed(0)
    T, H, I, r, s = 33, 64, 48, 4, 2.0
    x = torch.randn(T, H, dtype=torch.float64, requires_grad=True)
    wgu = torch.randn(2 * I, H, dtype=torch.float64)
    P = [torch.randn(*sh, dtype=torch.float64, requires_grad=True) for sh in ((r, H), (I, r), (r, H), (I, r))]
    Ag, Bg, Au, Bu = P
    g, u = lora_gate_up(x, wgu, Ag, Bg, Au, Bu, s)
    g_ref, u_ref = plain(x, wgu[:I], Ag, Bg, s), plain(x, wgu[I:], Au, Bu, s)
    assert torch.allclose(g, g_ref) and torch.allclose(u, u_ref)
    assert torch.allclose(lora_linear(x, wgu[:I], Ag, Bg, s), g_ref)
    cot = torch.randn(T, I, dtype=torch.float64)
    ga = torch.autograd.grad((g * cot).sum() + (u * cot).sum(), [x] + P)
    gb = torch.autograd.grad((g_ref * cot).sum() + (u_ref * cot).sum(), [x] + P)
    assert all(torch.allclose(a, b) for a, b in zip(ga, gb))
    g0, u0 = lora_gate_up(x, wgu, None, None, None, None, 0.0)   # no LoRA: just the base projections
    assert torch.allclose(g0, x @ wgu[:I].t()) and torch.allclose(u0, x @ wgu[I:].t())


def test_layer_weights_stack_gate_up_as_views():
    g, u = torch.randn(6, 4), torch.randn(6, 4)
    W = {"mlp.shared_expert.gate_proj.weight": g, "mlp.shared_expert.up_proj.weight": u, "x": torch.ones(1)}
    LW = LayerWeights(0, "full_attention", W, None)
    gu = LW.W["mlp.shared_expert.gate_up_proj.weight"]
    assert torch.equal(gu, torch.cat((g, u)))
    assert LW.W["mlp.shared_expert.gate_proj.weight"].data_ptr() == gu.data_ptr()
    assert torch.equal(LW.W["mlp.shared_expert.up_proj.weight"], u)
    assert "mlp.shared_expert.gate_up_proj.weight" not in W     # the caller's dict is not modified
