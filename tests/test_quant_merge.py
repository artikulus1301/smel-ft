"""int4 quantization of frozen experts (error report + forward closeness) and exact LoRA merge."""
import torch

from helpers import dims_of, layer_dict, layers_from_hf, randomize_lora_b
from smel.layer_runner import LayerMajorRunner, make_ctx, rmsnorm
from smel.lora import BlockLoRA
from smel.loss import LMHead
from smel.moe import moe_forward
from smel.pack import merge_experts, merge_mlp
from smel.quant import dequant_int4, quantize_int4, rel_err
from smel.stream import DictSource


def test_int4_roundtrip_exact_on_grid():
    torch.manual_seed(0)
    q = torch.randint(0, 16, (4, 8, 128))
    scale = torch.rand(4, 8, 2) + 0.5
    zero = torch.randn(4, 8, 2)
    scale, zero = scale.bfloat16().float(), zero.bfloat16().float()
    w = (q.reshape(4, 8, 2, 64).float() * scale[..., None] + zero[..., None]).reshape(4, 8, 128)
    pk = quantize_int4(w, 64)
    back = dequant_int4(pk["q"], pk["scale"], pk["zero"], torch.float32)
    assert rel_err(back, w) < 5e-3  # only bf16 rounding of scale/zero


def test_triton_dequant_matches_torch():
    from smel.quant import dequant_int4_torch
    torch.manual_seed(0)
    w = torch.randn(6, 96, 256, device="cuda")
    pk = quantize_int4(w, 64)
    a = dequant_int4(pk["q"], pk["scale"], pk["zero"], torch.float32)
    b = dequant_int4_torch(pk["q"], pk["scale"], pk["zero"], torch.float32)
    assert torch.allclose(a, b, atol=1e-6)
    a16 = dequant_int4(pk["q"][2:5], pk["scale"][2:5], pk["zero"][2:5], torch.bfloat16)
    assert rel_err(a16, b[2:5]) < 4e-3


def test_int4_expert_error_report(model32, capsys):
    d = dims_of(model32)
    errs = []
    for l in range(d.n_layers):
        t = layer_dict(model32, l)
        for name in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
            w = t[name]
            pk = quantize_int4(w, 64)
            errs.append((l, name.split(".")[-1], rel_err(dequant_int4(pk["q"], pk["scale"], pk["zero"],
                                                                     torch.float32), w)))
    with capsys.disabled():
        print("\nint4 g64 weight rel err per layer:", " ".join(f"L{l}:{n[:2]}={e:.3f}" for l, n, e in errs))
    assert max(e for *_, e in errs) < 0.12


def test_int4_forward_close_to_bf16(model32, seqs, capsys):
    d = dims_of(model32)
    torch.manual_seed(0)
    x = torch.randn(400, d.hidden, device="cuda")
    LW16 = layers_from_hf(model32, layers=[5])[5]
    LW4 = layers_from_hf(model32, layers=[5], frozen="int4")[5]
    with torch.no_grad():
        e_moe = rel_err(moe_forward(x, LW4.W, LW4.experts, d.top_k), moe_forward(x, LW16.W, LW16.experts, d.top_k))
    # whole model logits with every layer's experts frozen in int4
    head = LMHead.from_weight(model32.lm_head.weight.detach())
    outs = {}
    for mode in ("bf16", "int4"):
        run = LayerMajorRunner(d, DictSource(layers_from_hf(model32, frozen=mode)), list(range(d.n_layers)),
                               None, model32.model.norm.weight.detach(), head)
        hs = [model32.model.embed_tokens.weight.detach()[ids] for ids, _ in seqs]
        ctxs = [make_ctx([len(ids)], d, "cuda", torch.float32) for ids, _ in seqs]
        hs, _ = run.forward(hs, ctxs)
        outs[mode] = torch.cat([head.logits(rmsnorm(h, run.final_norm, d.eps)) for h in hs])
    e_logits = rel_err(outs["int4"], outs["bf16"])
    with capsys.disabled():
        # the random tiny model is chaotic (HF bf16 vs fp32 alone gives ~0.3 on logits), so the full-model number is
        # only reported; the real-model check is done when packing (stage 5)
        print(f"\nint4 frozen experts: MoE output rel err {e_moe:.3f}, full-model logits rel err {e_logits:.3f}")
    assert e_moe < 0.15


def test_merge_equals_base_plus_lora(model32):
    d = dims_of(model32)
    l = 3
    sel = {l: [1, 4, 9, 17]}
    lora = BlockLoRA.build(sel, d.hidden, d.moe_inter, d.shared_inter, r=8, alpha=16).cuda()
    randomize_lora_b(lora, std=0.05)
    LW = layers_from_hf(model32, layers=[l], selection=sel)[l]
    torch.manual_seed(0)
    x = torch.randn(300, d.hidden, device="cuda")
    with torch.no_grad():
        with_lora = moe_forward(x, LW.W, LW.experts, d.top_k, lora.get(l))
        t = layer_dict(model32, l)
        gu, dn = merge_experts(t["mlp.experts.gate_up_proj"], t["mlp.experts.down_proj"], lora.get(l).experts)
        p = "mlp.shared_expert."
        g, u, dd = merge_mlp(t[p + "gate_proj.weight"], t[p + "up_proj.weight"], t[p + "down_proj.weight"],
                             lora.get(l).shared)
        t2 = dict(t)
        t2.update({"mlp.experts.gate_up_proj": gu, "mlp.experts.down_proj": dn, p + "gate_proj.weight": g,
                   p + "up_proj.weight": u, p + "down_proj.weight": dd})
        from smel.moe import BF16Experts, LayerExperts
        merged = moe_forward(x, t2, LayerExperts(BF16Experts(gu, dn), group=8), d.top_k)
        base = moe_forward(x, LW.W, LW.experts, d.top_k)
    assert rel_err(with_lora, base) > 1e-2  # LoRA really changes the output
    assert rel_err(merged, with_lora) < 1e-5
