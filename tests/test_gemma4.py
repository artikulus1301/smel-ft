"""Gemma 4 MoE path == transformers Gemma4ForCausalLM on a tiny random model of the same class.

Sequences are longer than the tiny sliding window (16), so the window masks are exercised, with and without
sequence packing."""
import os

import pytest
import torch
import torch.nn.functional as F

from helpers import randomize_lora_b, seq_loss_sum
from smel.layer_runner import LayerMajorRunner, LayerWeights, make_ctx, rmsnorm
from smel.lora import BlockLoRA
from smel.loss import LMHead, chunked_ce
from smel.model_io import Checkpoint, Dims
from smel.moe import BF16Experts, FrozenExpertsFn, Int4Experts, LayerExperts
from smel.pack import merge_experts, merge_mlp
from smel.quant import rel_err
from smel.stream import DictSource
from smel.tiny import build_tiny_gemma4, save_tiny_checkpoint

DEV = "cuda"


@pytest.fixture(scope="module")
def g32():
    return build_tiny_gemma4(seed=0, dtype=torch.float32)


@pytest.fixture(scope="module")
def g16():
    return build_tiny_gemma4(seed=0, dtype=torch.bfloat16)


@pytest.fixture(scope="module")
def gseqs():
    g = torch.Generator().manual_seed(7)
    out = []
    for n in (23, 41, 66):
        ids = torch.randint(0, 512, (n,), generator=g)
        tgt = torch.full((n,), -100, dtype=torch.long)
        tgt[n // 3:n - 1] = ids[n // 3 + 1:n]
        out.append((ids.to(DEV), tgt.to(DEV)))
    return out


def dims(model):
    return Dims.from_config(model.config.to_dict())


def gemma_layers(model, layers, selection=None, frozen="bf16"):
    d = dims(model)
    out = {}
    for l in layers:
        p = f"model.layers.{l}."
        t = {k[len(p):]: v.detach() for k, v in model.state_dict().items() if k.startswith(p)}
        gu, dn = t.pop("experts.gate_up_proj"), t.pop("experts.down_proj")
        bank = BF16Experts(gu, dn) if frozen == "bf16" else Int4Experts.from_bf16(gu, dn, 16)
        sel = (selection or {}).get(l)
        if sel:
            ids = torch.as_tensor(sel, device=gu.device)
            ex = LayerExperts(bank, sel, gu[ids].clone(), dn[ids].clone(), group=4, act=d.act)
        else:
            ex = LayerExperts(bank, group=4, act=d.act)
        out[l] = LayerWeights(l, d.layer_types[l], t, ex)
    return out


def runner(model, layers, lora=None, selection=None, frozen="bf16"):
    d = dims(model)
    head = LMHead.from_weight(model.lm_head.weight.detach(), softcap=d.logit_softcap)
    return d, LayerMajorRunner(d, DictSource(gemma_layers(model, layers, selection, frozen)), layers, lora,
                               model.model.norm.weight.detach(), head)


def mbs(model, seqs, packing):
    d = dims(model)
    embed = model.model.embed_tokens.weight.detach()
    dt = embed.dtype
    groups = [[0, 1, 2]] if packing else [[0], [1], [2]]
    hs, ctxs, tg = [], [], []
    for g in groups:
        ids = torch.cat([seqs[i][0] for i in g])
        # same as Gemma4TextScaledWordEmbedding: scale cast to the weight dtype
        hs.append(embed[ids] * torch.tensor(model.config.hidden_size ** 0.5, dtype=dt, device=DEV))
        ctxs.append(make_ctx([len(seqs[i][0]) for i in g], d, DEV, dt))
        tg.append(torch.cat([seqs[i][1] for i in g]))
    return hs, ctxs, tg


def hf_logits(model, seqs):
    with torch.no_grad():
        return [model(input_ids=ids[None]).logits[0].float() for ids, _ in seqs]


# ------------------------------------------------------------------------------------------------- forward
def test_dims_and_ctx(g32):
    d = dims(g32)
    assert d.arch == "gemma4" and d.act == "gelu_tanh" and not d.norm_plus_one
    assert (d.global_head_dim, d.global_n_kv, d.global_rotary_dim, d.k_eq_v) == (64, 1, 16, True)
    ctx = make_ctx([10, 30], d, DEV, torch.float32)       # second sequence is longer than the window (16)
    assert ctx.sliding_mask is not None
    assert not ctx.sliding_mask[20, 5] and ctx.sliding_mask[20, 10]          # other sequence / same one, in window
    assert ctx.sliding_mask[39, 24] and not ctx.sliding_mask[39, 23]          # 39 - 23 = 16 -> outside the window
    assert not ctx.sliding_mask[24, 25]                                        # causal


@pytest.mark.parametrize("packing", [False, True])
def test_forward_logits_and_loss_fp32(g32, gseqs, packing):
    d, run = runner(g32, list(range(dims(g32).n_layers)))
    hs, ctxs, tg = mbs(g32, gseqs, packing)
    with torch.no_grad():
        out, _ = run.forward(hs, ctxs)
        ours = torch.cat([run.lm_head.logits(rmsnorm(h, run.final_norm, d.eps, d.norm_plus_one)) for h in out])
    ref = hf_logits(g32, gseqs)
    assert rel_err(ours, torch.cat(ref)) < 1e-4, rel_err(ours, torch.cat(ref))
    denom = sum(int((t != -100).sum()) for _, t in gseqs)
    ref_loss = float(sum(seq_loss_sum(lg, t) for lg, (_, t) in zip(ref, gseqs)) / denom)
    hs, ctxs, tg = mbs(g32, gseqs, packing)
    assert abs(run.eval_loss(hs, ctxs, tg, denom) - ref_loss) / ref_loss < 1e-5


def test_forward_bf16_not_worse_than_hf_bf16(g32, g16, gseqs):
    d, run = runner(g16, list(range(dims(g16).n_layers)))
    hs, ctxs, _ = mbs(g16, gseqs, True)
    with torch.no_grad():
        out, _ = run.forward(hs, ctxs)
        ours = torch.cat([run.lm_head.logits(rmsnorm(h, run.final_norm, d.eps, False)) for h in out])
    ref32, hf16 = torch.cat(hf_logits(g32, gseqs)), torch.cat(hf_logits(g16, gseqs))
    e_ours, e_hf = rel_err(ours, ref32), rel_err(hf16, ref32)
    print(f"bf16 logits err vs fp32: ours {e_ours:.4f}, HF {e_hf:.4f}")
    assert e_ours < 1.25 * e_hf + 1e-2


def test_int4_frozen_experts_close(g32, gseqs):
    """Per layer: the update a layer adds to the residual stream, int4 frozen experts vs bf16 (random weights
    compound over layers, so the whole-model drift of a tiny random model says little)."""
    d = dims(g32)
    hs, ctxs, _ = mbs(g32, gseqs, True)
    worst = 0.0
    for l in range(d.n_layers):
        _, r_bf = runner(g32, [l])
        _, r_q = runner(g32, [l], frozen="int4")
        with torch.no_grad():
            a, _ = r_bf.forward([hs[0].clone()], ctxs)
            b, _ = r_q.forward([hs[0].clone()], ctxs)
        worst = max(worst, rel_err(b[0] - hs[0], a[0] - hs[0]))
    print(f"int4 frozen experts: worst per-layer update rel err {worst:.3f}")
    assert worst < 0.2


# ------------------------------------------------------------------------------------------------- gradients
def test_frozen_experts_gelu_backward_matches_autograd():
    torch.manual_seed(0)
    E, H, I, N = 6, 32, 24, 50
    gu, dn = torch.randn(E, 2 * I, H, device=DEV) * 0.2, torch.randn(E, H, I, device=DEV) * 0.2
    counts = [9, 0, 12, 7, 15, 7]
    x = torch.randn(N, H, device=DEV, requires_grad=True)
    gy = torch.randn(N, H, device=DEV)
    FrozenExpertsFn.apply(x, counts, BF16Experts(gu, dn), 2, "gelu_tanh").backward(gy)
    ours = x.grad.clone()
    x.grad = None
    outs, s = [], 0
    for e, c in enumerate(counts):
        g, u = (x[s:s + c] @ gu[e].t()).chunk(2, -1)
        outs.append((F.gelu(g, approximate="tanh") * u) @ dn[e].t())
        s += c
    torch.cat(outs).backward(gy)
    assert rel_err(ours, x.grad) < 1e-5


def _patch_hf(model, lora, forced):
    """Reference: HF modules with LoRA on the selected experts + the dense MLP, and forced expert choices."""
    import types
    saved = []
    for l, layer in enumerate(model.model.layers):
        LL = lora.get(l)
        if LL is None:
            continue
        ex, mlp, router, q = layer.experts, layer.mlp, layer.router, list(forced[l])
        slot = {int(e): j for j, e in enumerate(LL.experts.expert_ids.tolist())} if LL.experts is not None else {}

        def efwd(self, hidden_states, top_k_index, top_k_weights, slot=slot, L=LL.experts):
            out = torch.zeros_like(hidden_states)
            for e in range(self.num_experts):
                tok, pos = torch.where(top_k_index == e)
                if tok.numel() == 0:
                    continue
                x = hidden_states[tok]
                I = self.down_proj.shape[2]
                g, u = x @ self.gate_up_proj[e, :I].t(), x @ self.gate_up_proj[e, I:].t()
                if e in slot:
                    j, s = slot[e], L.scale
                    g = g + s * (x @ L.A_gate[j].t()) @ L.B_gate[j].t()
                    u = u + s * (x @ L.A_up[j].t()) @ L.B_up[j].t()
                a = self.act_fn(g) * u
                y = a @ self.down_proj[e].t()
                if e in slot:
                    y = y + s * (a @ L.A_down[j].t()) @ L.B_down[j].t()
                out = out.index_add(0, tok, y * top_k_weights[tok, pos, None])
            return out

        def mfwd(self, x, L=LL.shared):
            s = L.scale
            g = self.gate_proj(x) + s * (x @ L.A_gate.t()) @ L.B_gate.t()
            u = self.up_proj(x) + s * (x @ L.A_up.t()) @ L.B_up.t()
            a = self.act_fn(g) * u
            return self.down_proj(a) + s * (a @ L.A_down.t()) @ L.B_down.t()

        def rfwd(self, hidden_states, q=q):
            hs = self.norm(hidden_states) * self.scale * self.scalar_root_size
            probs = torch.softmax(self.proj(hs), dim=-1, dtype=torch.float32)
            idx = q.pop(0)
            top = probs.gather(1, idx)
            top = top / top.sum(-1, keepdim=True)
            return probs, top * self.per_expert_scale[idx], idx

        for mod, fn in ((ex, efwd), (mlp, mfwd), (router, rfwd)):
            mod.forward = types.MethodType(fn, mod)
            saved.append(mod)
    return saved


@pytest.mark.parametrize("packing", [False, True])
def test_lora_grads_match_autograd(g32, gseqs, packing):
    model = g32
    d = dims(model)
    L = 2
    block = list(range(L, d.n_layers))
    selection = {l: [(3 * l) % 16, (5 * l + 1) % 16, (7 * l + 2) % 16] for l in block}
    torch.manual_seed(0)
    lora = BlockLoRA.build(selection, d.hidden, d.moe_inter, d.shared_inter, r=4, alpha=8).cuda()
    randomize_lora_b(lora)
    denom = sum(int((t != -100).sum()) for _, t in gseqs)
    _, low = runner(model, list(range(L)))
    _, run = runner(model, block, lora, selection)

    routes = {}
    cb = lambda l, mb, p, v, t: routes.setdefault(l, []).append(t)
    hs, ctxs, tg = mbs(model, gseqs, packing)
    hs, _ = low.forward(hs, ctxs)
    run.forward(hs, ctxs, stats_cb=cb)
    lens = [len(ids) for ids, _ in gseqs]
    forced = {l: [x for t in v for x in (t.split(lens) if packing else [t])] for l, v in routes.items()}

    saved = _patch_hf(model, lora, forced)
    try:
        loss = sum(seq_loss_sum(model(input_ids=ids[None]).logits[0], t) for ids, t in gseqs) / denom
        loss.backward()
    finally:
        for m in saved:
            del m.__dict__["forward"]
    ref = {n: p.grad.clone() for n, p in lora.named_parameters()}
    ref_loss = float(loss)
    lora.zero_grad(set_to_none=True)

    hs, ctxs, tg = mbs(model, gseqs, packing)
    hs, _ = low.forward(hs, ctxs)
    our_loss = run.train_step(hs, ctxs, tg, denom)
    assert abs(our_loss - ref_loss) / ref_loss < 1e-5
    worst = max(rel_err(p.grad, ref[n]) for n, p in lora.named_parameters())
    print(f"worst LoRA grad rel err {worst:.2e}")
    assert worst < 1e-4


def test_softcapped_chunked_ce():
    torch.manual_seed(0)
    V, H, N = 3000, 64, 120
    W = torch.randn(V, H, device=DEV) * 0.5
    head = LMHead.from_weight(W, vocab_chunk=700, softcap=30.0)
    h = (torch.randn(N, H, device=DEV) * 3).requires_grad_(True)
    tgt = torch.randint(0, V, (N,), device=DEV)
    loss = chunked_ce(h, head, tgt, denom=N, token_chunk=50)
    loss.backward()
    hr = h.detach().clone().requires_grad_(True)
    ref = F.cross_entropy(30.0 * torch.tanh((hr @ W.t()) / 30.0), tgt)
    ref.backward()
    assert abs(float(loss) - float(ref)) < 1e-4 and rel_err(h.grad, hr.grad) < 1e-4


# ------------------------------------------------------------------------------------------------- checkpoint
def test_checkpoint_names_embed_and_merge(g16, tmp_path):
    path = save_tiny_checkpoint(g16, str(tmp_path / "g4"), n_shards=2)
    ck = Checkpoint(path)
    assert ck.arch == "gemma4" and ck.lm_head_name() == ck.embed_name()       # tied embeddings
    t = ck.layer(2)
    assert "mlp.experts.gate_up_proj" in t and "experts.gate_up_proj" not in t and "layer_scalar" in t
    assert ck.full_name(2, "mlp.experts.gate_up_proj") == "model.language_model.layers.2.experts.gate_up_proj"
    assert [n.split(".")[-2] for n in ck.shared_mlp_names(2)] == ["gate_proj", "up_proj", "down_proj"]
    ids = torch.tensor([3, 7, 11])
    ref = g16.model.embed_tokens(ids.to(DEV)).cpu()
    assert torch.equal(ck.embed(ids), ref)
    # merge: base + delta of the selected experts / dense MLP
    lora = BlockLoRA.build({2: [1, 5]}, 256, 64, 128, r=4, alpha=8)
    randomize_lora_b(lora)
    LL = lora.get(2)
    gu, dn = merge_experts(t["mlp.experts.gate_up_proj"].float(), t["mlp.experts.down_proj"].float(), LL.experts)
    dg, du, dd = LL.experts.delta(1)
    assert torch.allclose(gu[5, :64], t["mlp.experts.gate_up_proj"][5, :64].float() + dg, atol=1e-6)
    assert torch.equal(gu[0], t["mlp.experts.gate_up_proj"][0].float())
    g, u, d_ = merge_mlp(*(t[n].float() for n in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")),
                         LL.shared)
    assert torch.allclose(d_, t["mlp.down_proj.weight"].float() + LL.shared.delta()[2], atol=1e-6)


@pytest.mark.skipif(not os.environ.get("SMEL_GEMMA4_TOKENIZER"), reason="set SMEL_GEMMA4_TOKENIZER (dir or hub id)")
def test_gemma4_chat_masks():
    from transformers import AutoTokenizer
    import numpy as np
    from smel.data import encode_chat
    tok = AutoTokenizer.from_pretrained(os.environ["SMEL_GEMMA4_TOKENIZER"])
    m = [{"role": "system", "content": "Ты Азуса."}, {"role": "user", "content": "Привет"},
         {"role": "assistant", "content": "Ну привет."}, {"role": "user", "content": "Как дела?"},
         {"role": "assistant", "content": "Нормально, отстань."}]
    ids, tgt = encode_chat(tok, m)
    sup = np.zeros(len(ids), dtype=bool)
    sup[1:] = tgt[:-1] != -100
    pieces, cur = [], []
    for i, s in enumerate(sup):
        if s:
            cur.append(int(ids[i]))
        elif cur:
            pieces.append(tok.decode(cur))
            cur = []
    assert pieces == ["Ну привет.<turn|>", "Нормально, отстань.<turn|>"]
    # the training text up to the last answer == the generation prompt for that answer (history without a thought
    # channel, the answer after an empty one), exactly what llama-server feeds with thinking off
    gen = tok.apply_chat_template(m[:4], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert tok.decode(ids).startswith(gen)
    assert tok.decode(ids).count("<|channel>") == 1
