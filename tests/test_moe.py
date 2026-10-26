"""Own MoE forward == transformers Qwen3_5MoeSparseMoeBlock (fp32 tight, bf16 within bf16 tolerance)."""
import pytest
import torch

from helpers import dims_of, layers_from_hf
from smel.moe import moe_forward
from smel.quant import rel_err


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("with_sel", [False, True])
def test_moe_matches_hf(model32, model16, dtype, tol, with_sel):
    model = model32 if dtype == torch.float32 else model16
    d = dims_of(model)
    sel = {1: [0, 3, 7, 30]} if with_sel else None
    LW = layers_from_hf(model, layers=[1], selection=sel)[1]
    torch.manual_seed(0)
    x = torch.randn(300, d.hidden, device="cuda", dtype=dtype)
    with torch.no_grad():
        ref = model.model.layers[1].mlp(x[None])[0]
        ours = moe_forward(x, LW.W, LW.experts, d.top_k)
    assert rel_err(ours, ref) < tol, rel_err(ours, ref)


def test_frozen_experts_backward_matches_autograd(model32):
    """Custom autograd of frozen experts == plain autograd through HF block (grad wrt input)."""
    d = dims_of(model32)
    LW = layers_from_hf(model32, layers=[2])[2]
    torch.manual_seed(1)
    x = torch.randn(200, d.hidden, device="cuda", requires_grad=True)
    gy = torch.randn(200, d.hidden, device="cuda")
    model32.model.layers[2].mlp(x[None])[0].backward(gy)
    ref = x.grad.clone()
    x.grad = None
    moe_forward(x, LW.W, LW.experts, d.top_k).backward(gy)
    assert rel_err(x.grad, ref) < 1e-5
