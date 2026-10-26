"""Shared helpers for tests: build our LayerWeights from an HF model and patch LoRA into the HF reference."""
import torch
import torch.nn.functional as F

from smel.model_io import Dims
from smel.moe import BF16Experts, Int4Experts, LayerExperts
from smel.layer_runner import LayerWeights


def dims_of(model):
    return Dims.from_config(model.config.to_dict())


def layer_dict(model, l):
    p = f"model.layers.{l}."
    return {k[len(p):]: v.detach() for k, v in model.state_dict().items() if k.startswith(p)}


def layers_from_hf(model, layers=None, selection=None, frozen="bf16", group=8, quant_group=64):
    d = dims_of(model)
    selection = selection or {}
    out = {}
    for l in (range(d.n_layers) if layers is None else layers):
        t = layer_dict(model, l)
        gu, dn = t.pop("mlp.experts.gate_up_proj"), t.pop("mlp.experts.down_proj")
        bank = BF16Experts(gu, dn) if frozen == "bf16" else Int4Experts.from_bf16(gu, dn, quant_group)
        sel = selection.get(l)
        if sel:
            ids = torch.as_tensor(sel, device=gu.device)
            ex = LayerExperts(bank, sel, gu[ids].clone(), dn[ids].clone(), group=group)
        else:
            ex = LayerExperts(bank, group=group)
        out[l] = LayerWeights(l, d.layer_types[l], t, ex)
    return out


def randomize_lora_b(lora, std=0.02, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for n, p in lora.named_parameters():
            if ".B_" in n or n.split(".")[-1].startswith("B_"):
                p.copy_(torch.randn(p.shape, generator=g) * std)


def patch_hf_lora(model, lora):
    """Monkeypatch HF expert / shared-expert modules so the HF model computes base + LoRA (reference autograd)."""
    for l, layer in enumerate(model.model.layers):
        LL = lora.get(l)
        if LL is None:
            continue
        experts, sh = layer.mlp.experts, layer.mlp.shared_expert
        if LL.experts is not None:
            slot = {int(e): j for j, e in enumerate(LL.experts.expert_ids.tolist())}

            def fwd(hidden_states, top_k_index, top_k_weights, experts=experts, slot=slot, L=LL.experts):
                final = torch.zeros_like(hidden_states)
                for e in range(experts.num_experts):
                    tok, pos = torch.where(top_k_index == e)
                    if tok.numel() == 0:
                        continue
                    x = hidden_states[tok]
                    gu, dn = experts.gate_up_proj[e], experts.down_proj[e]
                    I = dn.shape[1]
                    g, u = x @ gu[:I].t(), x @ gu[I:].t()
                    if e in slot:
                        j, s = slot[e], L.scale
                        g = g + s * (x @ L.A_gate[j].to(x.dtype).t()) @ L.B_gate[j].to(x.dtype).t()
                        u = u + s * (x @ L.A_up[j].to(x.dtype).t()) @ L.B_up[j].to(x.dtype).t()
                    a = F.silu(g) * u
                    y = a @ dn.t()
                    if e in slot:
                        y = y + s * (a @ L.A_down[j].to(x.dtype).t()) @ L.B_down[j].to(x.dtype).t()
                    final = final.index_add(0, tok, (y * top_k_weights[tok, pos, None]).to(final.dtype))
                return final

            experts.forward = fwd
        if LL.shared is not None:
            def sfwd(x, sh=sh, L=LL.shared):
                s = L.scale
                c = lambda t: t.to(x.dtype)
                g = sh.gate_proj(x) + s * (x @ c(L.A_gate).t()) @ c(L.B_gate).t()
                u = sh.up_proj(x) + s * (x @ c(L.A_up).t()) @ c(L.B_up).t()
                a = F.silu(g) * u
                return sh.down_proj(a) + s * (a @ c(L.A_down).t()) @ c(L.B_down).t()

            sh.forward = sfwd


def patch_hf_routing(model, forced: dict):
    """Force HF routers to pick the given expert indices (forced[l] = list of [T, K] tensors, one per HF call).
    Routing weights are still computed differentiably from the router softmax, exactly like the original."""
    for l, layer in enumerate(model.model.layers):
        if l not in forced:
            continue
        router, queue = layer.mlp.gate, list(forced[l])

        def rfwd(hidden_states, mod=router, q=queue):
            hidden_states = hidden_states.reshape(-1, mod.hidden_dim)
            logits = F.linear(hidden_states, mod.weight)
            probs = torch.softmax(logits, dtype=torch.float, dim=-1)
            idx = q.pop(0)
            top = probs.gather(1, idx)
            top = (top / top.sum(dim=-1, keepdim=True)).to(logits.dtype)
            return logits, top, idx

        router.forward = rfwd


def unpatch_hf(model):
    for layer in model.model.layers:
        for m in (layer.mlp.experts, layer.mlp.shared_expert, layer.mlp.gate):
            if "forward" in m.__dict__:
                del m.__dict__["forward"]


def hf_hidden_at(model, ids, layer):
    """Input hidden state of `layer` (= output of layer-1) for one sequence, via HF forward."""
    out = model.model(input_ids=ids[None], output_hidden_states=True)
    return out.hidden_states[layer][0]


def seq_loss_sum(logits, tgt):
    m = tgt != -100
    return F.cross_entropy(logits[m].float(), tgt[m].long(), reduction="sum")
