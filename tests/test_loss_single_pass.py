"""Single-pass chunked CE (smel/loss.py _ce_chunk) against full logits in fp32 (with and without final softcapping) and
against the two-pass reference in bf16; plus the int8 head."""
import pytest
import torch

from smel.loss import LMHead, _ce_chunk, _ce_chunk_two_pass, chunked_ce
from smel.quant import quantize_int8_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def full_ref(h, W, tgt, softcap, denom):
    h = h.detach().clone().requires_grad_(True)
    lg = (h @ W.float().t())
    if softcap:
        lg = softcap * torch.tanh(lg / softcap)
    loss = torch.nn.functional.cross_entropy(lg, tgt, reduction="sum") / denom
    loss.backward()
    return loss.detach(), h.grad


@pytest.mark.parametrize("softcap", [0.0, 30.0])
def test_fp32_matches_full_logits(softcap):
    torch.manual_seed(0)
    n, H, V = 37, 64, 1000                      # V is not a multiple of the vocab chunk
    h = torch.randn(n, H, device="cuda") * 0.5
    W = torch.randn(V, H, device="cuda") * 0.2
    tgt = torch.randint(0, V, (n,), device="cuda")
    head = LMHead(weight=W, vocab_chunk=128, softcap=softcap)
    loss_ref, grad_ref = full_ref(h, W, tgt, softcap, denom=n)
    h_ = h.clone().requires_grad_(True)
    loss = chunked_ce(h_, head, tgt, denom=n, token_chunk=16)
    loss.backward()
    assert abs(loss.item() - loss_ref.item()) < 1e-4 * max(1.0, abs(loss_ref.item()))
    assert (h_.grad - grad_ref).norm() / grad_ref.norm() < 1e-4


def test_bf16_matches_two_pass():
    torch.manual_seed(1)
    n, H, V = 300, 128, 5000
    h = (torch.randn(n, H, device="cuda") * 0.5).bfloat16()
    W = (torch.randn(V, H, device="cuda") * 0.2).bfloat16()
    tgt = torch.randint(0, V, (n,), device="cuda")
    for softcap in (0.0, 30.0):
        head = LMHead(weight=W, vocab_chunk=1024, softcap=softcap)
        l1, g1 = _ce_chunk(h, head, tgt, True)
        l2, g2 = _ce_chunk_two_pass(h, head, tgt, True)
        assert abs(l1.item() - l2.item()) / abs(l2.item()) < 1e-3
        assert (g1 - g2).norm() / g2.norm() < 2e-2     # bf16 probabilities in both; different accumulation order


def test_int8_head_runs_and_matches_its_dequantized_weight():
    torch.manual_seed(2)
    n, H, V = 64, 96, 700
    h = torch.randn(n, H, device="cuda") * 0.5
    W = torch.randn(V, H, device="cuda") * 0.2
    tgt = torch.randint(0, V, (n,), device="cuda")
    from smel.quant import dequant_int8_rows
    q = {k: v.cuda() for k, v in quantize_int8_rows(W).items()}
    w_dq = dequant_int8_rows(q["q"], q["scale"], torch.float32)     # what the head actually multiplies by
    loss_q = chunked_ce(h, LMHead(int8=q, vocab_chunk=256), tgt, denom=n, token_chunk=32, need_grad=False)
    loss_dq = chunked_ce(h, LMHead(weight=w_dq), tgt, denom=n, token_chunk=32, need_grad=False)
    assert abs(loss_q.item() - loss_dq.item()) < 1e-3 * abs(loss_dq.item())
