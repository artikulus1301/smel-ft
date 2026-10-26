"""Layer-major forward/backward over packed microbatches == ordinary HF forward / autograd."""
import pytest
import torch

from helpers import (dims_of, layers_from_hf, patch_hf_lora, patch_hf_routing, randomize_lora_b, seq_loss_sum,
                     unpatch_hf)
from smel.layer_runner import LayerMajorRunner, make_ctx, rmsnorm
from smel.lora import BlockLoRA
from smel.loss import LMHead
from smel.quant import rel_err
from smel.stream import DictSource


def _mbs(seqs, d, dtype, embed, packing):
    groups = [[0, 1, 2]] if packing else [[0], [1], [2]]
    hs, ctxs, tg = [], [], []
    for g in groups:
        ids = torch.cat([seqs[i][0] for i in g])
        hs.append(embed[ids].to(dtype))
        ctxs.append(make_ctx([len(seqs[i][0]) for i in g], d, "cuda", dtype))
        tg.append(torch.cat([seqs[i][1] for i in g]))
    return hs, ctxs, tg


def _runner(model, layers, lora=None, selection=None):
    d = dims_of(model)
    src = DictSource(layers_from_hf(model, layers=layers, selection=selection))
    head = LMHead.from_weight(model.lm_head.weight.detach())
    return d, LayerMajorRunner(d, src, layers, lora, model.model.norm.weight.detach(), head)


def _our_logits_and_loss(model, seqs, packing):
    d, run = _runner(model, list(range(dims_of(model).n_layers)))
    dtype = model.lm_head.weight.dtype
    embed = model.model.embed_tokens.weight.detach()
    denom = sum(int((t != -100).sum()) for _, t in seqs)
    hs, ctxs, tg = _mbs(seqs, d, dtype, embed, packing)
    with torch.no_grad():
        out, _ = run.forward(hs, ctxs)
        logits = torch.cat([run.lm_head.logits(rmsnorm(h, run.final_norm, d.eps)) for h in out])
    hs, ctxs, tg = _mbs(seqs, d, dtype, embed, packing)
    return logits, run.eval_loss(hs, ctxs, tg, denom), denom


def _hf_logits_and_loss(model, seqs, denom):
    with torch.no_grad():
        logits = [model(input_ids=ids[None]).logits[0].float() for ids, _ in seqs]
    loss = sum(seq_loss_sum(lg, t) for lg, (_, t) in zip(logits, seqs)) / denom
    return torch.cat(logits), float(loss)


@pytest.mark.parametrize("packing", [False, True])
def test_forward_logits_and_loss_fp32(model32, seqs, packing):
    ours, our_loss, denom = _our_logits_and_loss(model32, seqs, packing)
    ref, ref_loss = _hf_logits_and_loss(model32, seqs, denom)
    assert rel_err(ours, ref) < 1e-3, rel_err(ours, ref)
    assert abs(our_loss - ref_loss) / ref_loss < 1e-4


@pytest.mark.parametrize("packing", [False, True])
def test_forward_bf16_not_worse_than_hf_bf16(model32, model16, seqs, packing):
    """bf16 noise of the random tiny model is large (HF bf16 vs fp32 ~0.3 on logits); our bf16 forward must be
    no further from the fp32 reference than HF's own bf16 forward."""
    ours, our_loss, denom = _our_logits_and_loss(model16, seqs, packing)
    ref32, ref32_loss = _hf_logits_and_loss(model32, seqs, denom)
    hf16, hf16_loss = _hf_logits_and_loss(model16, seqs, denom)
    e_ours, e_hf = rel_err(ours, ref32), rel_err(hf16, ref32)
    print(f"bf16 logits err vs fp32: ours {e_ours:.3f}, HF {e_hf:.3f}")
    assert e_ours < 1.25 * e_hf + 1e-2
    assert abs(our_loss - ref32_loss) <= 1.25 * abs(hf16_loss - ref32_loss) + 1e-2


@pytest.fixture(params=["torch_gdr", "fla"])
def gdr_impl(request, monkeypatch):
    """torch_gdr: exact fp32 delta rule on both sides (checks our machinery to ~1e-5);
    fla: FLA kernels on both sides (their fp32 path uses TF32 dots internally -> ~1e-3)."""
    import smel.layer_runner as LR
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M
    if request.param == "torch_gdr":
        monkeypatch.setattr(LR, "USE_FLA", False)
        monkeypatch.setattr(M, "torch_chunk_gated_delta_rule", LR._torch_reference_gdr())
        return 1e-4
    return 1e-2


@pytest.mark.parametrize("packing", [False, True])
def test_lora_grads_match_autograd(model32, seqs, packing, gdr_impl):
    tol = gdr_impl
    model = model32
    d = dims_of(model)
    L = 2
    block = list(range(L, d.n_layers))
    selection = {l: [(3 * l) % 32, (5 * l + 1) % 32, (7 * l + 2) % 32] for l in block}
    torch.manual_seed(0)
    lora = BlockLoRA.build(selection, d.hidden, d.moe_inter, d.shared_inter, r=4, alpha=8).cuda()
    randomize_lora_b(lora)
    denom = sum(int((t != -100).sum()) for _, t in seqs)
    _, low = _runner(model, list(range(L)))
    _, run = _runner(model, block, lora, selection)
    embed = model.model.embed_tokens.weight.detach()

    # capture our expert choices (with LoRA active) and force them in the reference, so that numerically
    # near-tied top-k decisions cannot flip between the two implementations
    routes = {}
    cb = lambda l, mb, p, v, t: routes.setdefault(l, []).append(t)
    hs, ctxs, tg = _mbs(seqs, d, torch.float32, embed, packing)
    hs, _ = low.forward(hs, ctxs, stats_cb=cb)
    run.forward(hs, ctxs, stats_cb=cb)
    lens = [len(ids) for ids, _ in seqs]
    forced = {l: [x for t in v for x in (t.split(lens) if packing else [t])] for l, v in routes.items()}

    # reference: full HF model with LoRA patched in, plain autograd
    patch_hf_lora(model, lora)
    patch_hf_routing(model, forced)
    try:
        loss = sum(seq_loss_sum(model(input_ids=ids[None]).logits[0], t) for ids, t in seqs) / denom
        loss.backward()
    finally:
        unpatch_hf(model)
    ref = {n: p.grad.clone() for n, p in lora.named_parameters()}
    ref_loss = float(loss)
    lora.zero_grad(set_to_none=True)

    # ours: layers 0..L-1 without grad (cache), then layer-major step on L..N-1
    hs, ctxs, tg = _mbs(seqs, d, torch.float32, embed, packing)
    hs, _ = low.forward(hs, ctxs)
    our_loss = run.train_step(hs, ctxs, tg, denom)
    assert abs(our_loss - ref_loss) / ref_loss < 1e-5
    worst = 0.0
    for n, p in lora.named_parameters():
        e = rel_err(p.grad, ref[n])
        worst = max(worst, e)
        assert e < tol, (n, e)
    print(f"worst LoRA grad rel err {worst:.2e}")


def test_cache_then_block_equals_full(model16, seqs, tmp_path):
    """Loss of 'cached input of layer L + block L..N' == loss of the full model; the cache goes through a bf16 memmap."""
    import numpy as np
    model = model16
    d = dims_of(model)
    L = 4
    embed = model.model.embed_tokens.weight.detach()
    denom = sum(int((t != -100).sum()) for _, t in seqs)
    _, full = _runner(model, list(range(d.n_layers)))
    _, low = _runner(model, list(range(L)))
    _, top = _runner(model, list(range(L, d.n_layers)))
    hs, ctxs, tg = _mbs(seqs, d, torch.bfloat16, embed, False)
    full_loss = full.eval_loss(hs, ctxs, tg, denom)
    hs, ctxs, tg = _mbs(seqs, d, torch.bfloat16, embed, False)
    with torch.no_grad():
        hs, _ = low.forward(hs, ctxs)
    # write to memmap and read back
    allh = torch.cat(hs).cpu()
    mm = np.lib.format.open_memmap(str(tmp_path / "h.npy"), mode="w+", dtype=np.uint16, shape=tuple(allh.shape))
    mm[:] = allh.view(torch.uint16).numpy()
    mm.flush()
    back = torch.from_numpy(np.load(str(tmp_path / "h.npy"), mmap_mode="r")[:].copy()).view(torch.bfloat16)
    assert torch.equal(back, allh)
    sizes = [h.shape[0] for h in hs]
    hs2 = [x.cuda() for x in back.split(sizes)]
    cache_loss = top.eval_loss(hs2, ctxs, tg, denom)
    assert abs(cache_loss - full_loss) / full_loss < 1e-6
    ref_loss = sum(seq_loss_sum(model(input_ids=ids[None]).logits[0], t) for ids, t in seqs) / denom
    assert abs(cache_loss - float(ref_loss)) / float(ref_loss) < 2e-2
