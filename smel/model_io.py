"""Layer-wise access to Qwen3.5/3.6-MoE safetensors checkpoints through mmap (safe_open) and model dimensions."""
import json
import os
from dataclasses import dataclass

import torch
from safetensors import safe_open


@dataclass
class Dims:
    hidden: int
    n_layers: int
    layer_types: list
    eps: float
    vocab: int
    # full attention
    n_heads: int
    n_kv: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    # gated deltanet
    lin_k_heads: int
    lin_v_heads: int
    lin_k_dim: int
    lin_v_dim: int
    conv_k: int
    # moe
    n_experts: int
    top_k: int
    moe_inter: int
    shared_inter: int

    @classmethod
    def from_config(cls, cfg: dict) -> "Dims":
        cfg = cfg.get("text_config", cfg)
        rope = cfg.get("rope_parameters") or {}
        prf = rope.get("partial_rotary_factor", cfg.get("partial_rotary_factor", 1.0))
        hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        return cls(
            hidden=cfg["hidden_size"], n_layers=cfg["num_hidden_layers"], layer_types=list(cfg["layer_types"]),
            eps=cfg["rms_norm_eps"], vocab=cfg["vocab_size"],
            n_heads=cfg["num_attention_heads"], n_kv=cfg["num_key_value_heads"], head_dim=hd,
            rotary_dim=int(hd * prf), rope_theta=rope.get("rope_theta", cfg.get("rope_theta", 10000.0)),
            lin_k_heads=cfg["linear_num_key_heads"], lin_v_heads=cfg["linear_num_value_heads"],
            lin_k_dim=cfg["linear_key_head_dim"], lin_v_dim=cfg["linear_value_head_dim"],
            conv_k=cfg["linear_conv_kernel_dim"],
            n_experts=cfg["num_experts"], top_k=cfg["num_experts_per_tok"], moe_inter=cfg["moe_intermediate_size"],
            shared_inter=cfg["shared_expert_intermediate_size"],
        )

    def expert_params(self) -> int:
        return 3 * self.hidden * self.moe_inter


class Checkpoint:
    """Read-only view of a (possibly sharded) safetensors checkpoint. Tensors are read lazily from mmap."""

    def __init__(self, model_dir: str):
        self.dir = model_dir
        idx = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            fname = "model.safetensors"
            with safe_open(os.path.join(model_dir, fname), "pt") as h:
                self.weight_map = {k: fname for k in h.keys()}
        with open(os.path.join(model_dir, "config.json")) as f:
            self.config = json.load(f)
        self.dims = Dims.from_config(self.config)
        if any(k.startswith("model.language_model.") for k in self.weight_map):
            self.prefix = "model.language_model."
        else:
            self.prefix = "model."
        self._handles = {}

    def _h(self, fname):
        # keep only the two most recent shards mapped: on Windows every open mapping counts against the commit
        # limit (RAM + pagefile), and 26 mapped shards (72 GB) exceed it -> os error 1455
        h = self._handles.pop(fname, None)
        if h is None:
            h = safe_open(os.path.join(self.dir, fname), "pt")
        self._handles[fname] = h
        while len(self._handles) > 2:
            self._handles.pop(next(iter(self._handles)))
        return h

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        return self._h(self.weight_map[name]).get_tensor(name)

    def get_rows(self, name: str, rows) -> torch.Tensor:
        """Read only the given indices along dim 0 (e.g. a few experts of a fused 3D tensor)."""
        sl = self._h(self.weight_map[name]).get_slice(name)
        return torch.stack([sl[int(r): int(r) + 1][0] for r in rows])

    def layer_names(self, l: int):
        p = f"{self.prefix}layers.{l}."
        return sorted(k for k in self.weight_map if k.startswith(p))

    def layer(self, l: int, skip_experts: bool = False) -> dict:
        """All tensors of decoder layer l with the 'layers.{l}.' prefix stripped."""
        p = f"{self.prefix}layers.{l}."
        out = {}
        for k in self.layer_names(l):
            short = k[len(p):]
            if skip_experts and short.startswith("mlp.experts."):
                continue
            out[short] = self.get(k)
        return out

    def embed_name(self):
        return f"{self.prefix}embed_tokens.weight"

    def norm_name(self):
        return f"{self.prefix}norm.weight"

    def lm_head_name(self):
        return "lm_head.weight" if self.has("lm_head.weight") else self.embed_name()

    def close(self):
        self._handles.clear()
