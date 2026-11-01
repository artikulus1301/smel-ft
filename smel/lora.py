"""LoRA parameters for selected routed experts (gate/up/down per expert) and for the shared expert of a layer."""
import math

import torch
from torch import nn


def _init_a(t: torch.Tensor):
    # same as PEFT: kaiming_uniform_(a=sqrt(5)) on each [r, in] matrix
    for m in t.reshape(-1, *t.shape[-2:]):
        nn.init.kaiming_uniform_(m, a=math.sqrt(5))


class ExpertsLoRA(nn.Module):
    """LoRA for k selected experts of one layer. Slot j corresponds to global expert id `expert_ids[j]`."""

    def __init__(self, expert_ids, hidden: int, inter: int, r: int = 16, alpha: float = 32.0):
        super().__init__()
        k = len(expert_ids)
        self.register_buffer("expert_ids", torch.as_tensor(list(expert_ids), dtype=torch.long), persistent=True)
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        self.A_gate = nn.Parameter(torch.empty(k, r, hidden))
        self.B_gate = nn.Parameter(torch.zeros(k, inter, r))
        self.A_up = nn.Parameter(torch.empty(k, r, hidden))
        self.B_up = nn.Parameter(torch.zeros(k, inter, r))
        self.A_down = nn.Parameter(torch.empty(k, r, inter))
        self.B_down = nn.Parameter(torch.zeros(k, hidden, r))
        for a in (self.A_gate, self.A_up, self.A_down):
            _init_a(a.data)

    def delta(self, j: int):
        """Dense weight deltas of slot j: (d_gate [I,H], d_up [I,H], d_down [H,I]) in fp32."""
        s = self.scale
        return (s * self.B_gate[j].float() @ self.A_gate[j].float(),
                s * self.B_up[j].float() @ self.A_up[j].float(),
                s * self.B_down[j].float() @ self.A_down[j].float())


class MLPLoRA(nn.Module):
    """LoRA for a plain gated MLP (the shared expert)."""

    def __init__(self, hidden: int, inter: int, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        self.A_gate = nn.Parameter(torch.empty(r, hidden))
        self.B_gate = nn.Parameter(torch.zeros(inter, r))
        self.A_up = nn.Parameter(torch.empty(r, hidden))
        self.B_up = nn.Parameter(torch.zeros(inter, r))
        self.A_down = nn.Parameter(torch.empty(r, inter))
        self.B_down = nn.Parameter(torch.zeros(hidden, r))
        for a in (self.A_gate, self.A_up, self.A_down):
            _init_a(a.data)

    def delta(self):
        s = self.scale
        return (s * self.B_gate.float() @ self.A_gate.float(),
                s * self.B_up.float() @ self.A_up.float(),
                s * self.B_down.float() @ self.A_down.float())


def lora_linear(x, w, A, B, scale):
    """x @ w.T + scale * (x @ A.T) @ B.T, with LoRA factors cast to the activation dtype. The scaled add is the
    epilogue of the second LoRA GEMM (addmm: no separate pass over y, and the LoRA term is not rounded before it)."""
    y = x @ w.t()
    if A is None:
        return y
    if x.dim() == 2:
        return torch.addmm(y, x @ A.to(x.dtype).t(), B.to(x.dtype).t(), alpha=scale)
    return y + scale * ((x @ A.to(x.dtype).t()) @ B.to(x.dtype).t())


def lora_gate_up(x, w_gate_up, A_gate, B_gate, A_up, B_up, scale):
    """(gate, up) of a gated MLP whose two projections read the same x: one base GEMM over the stacked [2I, H]
    weight and one GEMM over the stacked LoRA A [2r, H] instead of two each. x: [T, H]; A_*: [r, H]; B_*: [I, r]."""
    I = w_gate_up.shape[0] // 2
    base = x @ w_gate_up.t()
    if A_gate is None:
        return base[:, :I], base[:, I:]
    r = A_gate.shape[0]
    xa = x @ torch.cat((A_gate, A_up)).to(x.dtype).t()
    g = torch.addmm(base[:, :I], xa[:, :r], B_gate.to(x.dtype).t(), alpha=scale)
    u = torch.addmm(base[:, I:], xa[:, r:], B_up.to(x.dtype).t(), alpha=scale)
    return g, u


class LayerLoRA(nn.Module):
    def __init__(self, experts: ExpertsLoRA | None, shared: MLPLoRA | None):
        super().__init__()
        self.experts = experts
        self.shared = shared


class BlockLoRA(nn.Module):
    """All LoRA modules of the trained block, keyed by absolute layer index."""

    def __init__(self, layers: dict):
        super().__init__()
        self.layers = nn.ModuleDict({str(k): v for k, v in layers.items()})

    def get(self, l: int):
        return self.layers[str(l)] if str(l) in self.layers else None

    @classmethod
    def load(cls, path, dims, device="cuda"):
        """Adapter folder written by scripts/train.py -> (BlockLoRA, meta)."""
        import json
        import os
        from safetensors.torch import load_file
        meta = json.load(open(os.path.join(path, "adapter_config.json")))
        experts = {int(k): v for k, v in meta["experts"].items()}
        lora = cls.build(experts, dims.hidden, dims.moe_inter, dims.shared_inter, meta["r"], meta["alpha"],
                         shared=meta["shared_lora"])
        lora.load_state_dict(load_file(os.path.join(path, "adapter.safetensors")))
        return lora.to(device), meta

    @classmethod
    def build(cls, selection: dict, hidden: int, inter: int, shared_inter: int, r: int, alpha: float,
              shared: bool = True):
        """selection: {layer: [expert ids]}"""
        layers = {}
        for l, ids in selection.items():
            ex = ExpertsLoRA(ids, hidden, inter, r, alpha) if len(ids) else None
            sh = MLPLoRA(hidden, shared_inter, r, alpha) if shared else None
            layers[int(l)] = LayerLoRA(ex, sh)
        return cls(layers)
