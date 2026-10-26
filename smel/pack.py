"""Packing of the trained block (per-layer safetensors: dense bf16, frozen experts int4, selected experts bf16) and
exact merging of LoRA deltas into bf16 weights."""
import torch

from .quant import quantize_int4


def pack_layer(t: dict, sel_ids=None, group: int = 64, dtype=torch.bfloat16) -> dict:
    """t: tensors of one layer as stored in the checkpoint (names relative to the layer)."""
    out = {}
    for k, v in t.items():
        if k.startswith("mlp.experts."):
            continue
        keep = k.endswith(("A_log", "dt_bias")) or not v.is_floating_point()
        out[k] = v.contiguous() if keep else v.to(dtype).contiguous()
    gu, dn = t["mlp.experts.gate_up_proj"], t["mlp.experts.down_proj"]
    if sel_ids is not None and len(sel_ids):
        ids = torch.as_tensor(list(sel_ids), dtype=torch.long)
        out["experts.sel.ids"] = ids
        out["experts.sel.gate_up"] = gu[ids].to(dtype).contiguous()
        out["experts.sel.down"] = dn[ids].to(dtype).contiguous()
    for name, w in (("gate_up", gu), ("down", dn)):
        for kk, vv in quantize_int4(w, group).items():
            out[f"experts.{name}.{kk}"] = vv
    return out


@torch.no_grad()
def merge_experts(gate_up: torch.Tensor, down: torch.Tensor, lora) -> tuple:
    """Return copies of fused expert tensors with the LoRA deltas of the selected experts added (fp32 math)."""
    gu, dn = gate_up.clone(), down.clone()
    I = dn.shape[2]
    for j, e in enumerate(lora.expert_ids.tolist()):
        dg, du, dd = (x.to(gu.device) for x in lora.delta(j))
        gu[e, :I] = (gu[e, :I].float() + dg).to(gu.dtype)
        gu[e, I:] = (gu[e, I:].float() + du).to(gu.dtype)
        dn[e] = (dn[e].float() + dd).to(dn.dtype)
    return gu, dn


@torch.no_grad()
def merge_mlp(gate_w, up_w, down_w, lora) -> tuple:
    dg, du, dd = (x.to(gate_w.device) for x in lora.delta())
    return ((gate_w.float() + dg).to(gate_w.dtype), (up_w.float() + du).to(up_w.dtype),
            (down_w.float() + dd).to(down_w.dtype))
