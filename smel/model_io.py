"""Layer-wise access to MoE safetensors checkpoints (plain reads of single tensors) and model dimensions.

Supported architectures (`Dims.arch`):
  * qwen3_5_moe - Qwen3.5/3.6-MoE: Gated DeltaNet + gated full attention, 256 experts + shared expert.
  * gemma4      - Gemma 4 MoE (e.g. 26B-A4B): sliding-window + global attention, dense MLP in parallel with
                  128 experts.

Tensor names of the checkpoint are mapped to one internal layout per layer: routed experts are always
'mlp.experts.gate_up_proj' / 'mlp.experts.down_proj' (Gemma 4 stores them as 'experts.*'); everything else keeps
its checkpoint name. `Checkpoint.full_name` maps back for writing merged weights.
"""
import json
import os
import struct
from dataclasses import dataclass, field

import torch

ARCHS = ("qwen3_5_moe", "gemma4")


@dataclass
class Dims:
    hidden: int
    n_layers: int
    layer_types: list
    eps: float
    vocab: int
    # attention (Qwen: the full-attention layers; Gemma 4: the sliding-window layers)
    n_heads: int
    n_kv: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    # moe
    n_experts: int
    top_k: int
    moe_inter: int
    shared_inter: int          # Qwen: shared expert; Gemma 4: the dense MLP that runs in parallel with the experts
    arch: str = "qwen3_5_moe"
    act: str = "silu"          # expert / MLP activation: silu | gelu_tanh
    norm_plus_one: bool = True  # RMSNorm scale: x * (1 + w) (Qwen3.5) or x * w (Gemma 4)
    # gated deltanet (qwen3_5_moe)
    lin_k_heads: int = 0
    lin_v_heads: int = 0
    lin_k_dim: int = 0
    lin_v_dim: int = 0
    conv_k: int = 0
    # gemma4
    sliding_window: int = 0
    global_head_dim: int = 0
    global_n_kv: int = 0
    global_rotary_dim: int = 0  # rotated dims of the global layers (partial "proportional" RoPE)
    global_rope_theta: float = 0.0
    k_eq_v: bool = False        # global layers use the keys as values (no v_proj)
    logit_softcap: float = 0.0  # final logits = cap * tanh(logits / cap); 0 = off
    embed_scale: float = 1.0    # input embeddings are multiplied by this (Gemma: sqrt(hidden) rounded to bf16)
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict) -> "Dims":
        top = cfg
        cfg = cfg.get("text_config", cfg)
        mt = cfg.get("model_type") or top.get("model_type", "")
        if mt.startswith("gemma4"):
            return cls._gemma4(cfg)
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

    @classmethod
    def _gemma4(cls, cfg: dict) -> "Dims":
        if not cfg.get("enable_moe_block", False):
            raise ValueError("SMEL needs a Gemma 4 model with a MoE block (e.g. gemma-4-26B-A4B)")
        if cfg.get("hidden_size_per_layer_input") or cfg.get("num_kv_shared_layers"):
            raise ValueError("per-layer inputs / shared KV layers (Gemma 4 E-series) are not supported")
        rope = cfg["rope_parameters"]
        loc, glob = rope["sliding_attention"], rope["full_attention"]
        if loc.get("rope_type", "default") != "default" or glob.get("rope_type") != "proportional":
            raise ValueError(f"unsupported Gemma 4 RoPE setup: {rope}")
        # global-layer overrides: config.json has global_head_dim / num_global_key_value_heads, a config saved by
        # transformers (to_dict) has them per layer in per_layer_config instead
        plc = next((v for k, v in (cfg.get("per_layer_config") or {}).items()
                    if cfg["layer_types"][int(k)] == "full_attention"), {})
        ghd = cfg.get("global_head_dim") or plc.get("head_dim") or cfg["head_dim"]
        g_kv = cfg.get("num_global_key_value_heads") or plc.get("num_key_value_heads") or cfg["num_key_value_heads"]
        act = cfg.get("hidden_activation", "gelu_pytorch_tanh")
        if act not in ("gelu_pytorch_tanh", "gelu_tanh"):
            raise ValueError(f"unsupported activation {act}")
        # the reference casts sqrt(hidden) to the weight dtype (bf16) before multiplying
        emb_scale = float(torch.tensor(cfg["hidden_size"] ** 0.5, dtype=torch.bfloat16))
        return cls(
            arch="gemma4", act="gelu_tanh", norm_plus_one=False,
            hidden=cfg["hidden_size"], n_layers=cfg["num_hidden_layers"], layer_types=list(cfg["layer_types"]),
            eps=cfg["rms_norm_eps"], vocab=cfg["vocab_size"],
            n_heads=cfg["num_attention_heads"], n_kv=cfg["num_key_value_heads"], head_dim=cfg["head_dim"],
            rotary_dim=cfg["head_dim"], rope_theta=loc["rope_theta"],
            sliding_window=cfg["sliding_window"], global_head_dim=ghd,
            global_n_kv=g_kv,
            global_rotary_dim=2 * int(glob.get("partial_rotary_factor", 1.0) * ghd // 2),
            global_rope_theta=glob["rope_theta"], k_eq_v=bool(cfg.get("attention_k_eq_v", False)),
            logit_softcap=float(cfg.get("final_logit_softcapping") or 0.0), embed_scale=emb_scale,
            n_experts=cfg["num_experts"], top_k=cfg["top_k_experts"], moe_inter=cfg["moe_intermediate_size"],
            shared_inter=cfg["intermediate_size"],
        )

    def expert_params(self) -> int:
        return 3 * self.hidden * self.moe_inter


# internal layer-relative name -> checkpoint name, per architecture (only names that differ)
_RENAME = {
    "gemma4": {"mlp.experts.gate_up_proj": "experts.gate_up_proj", "mlp.experts.down_proj": "experts.down_proj"},
}
# the "shared" MLP that gets LoRA next to the experts: (gate, up, down) weight names, layer-relative
SHARED_MLP = {
    "qwen3_5_moe": ("mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
                    "mlp.shared_expert.down_proj.weight"),
    "gemma4": ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"),
}


_ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
              "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
              "BOOL": torch.bool}


class _Shard:
    """One safetensors file read with plain seek/read: only the bytes of the requested tensor (or rows) are read.

    safetensors.safe_open maps the whole file; on Windows a mapping counts against the commit limit
    (RAM + pagefile), so a single 50 GB shard (Gemma 4 26B) crashes the process with an access violation."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.hdr = json.loads(f.read(n))
        self.hdr.pop("__metadata__", None)
        self.base = 8 + n

    def keys(self):
        return list(self.hdr)

    def shape(self, name):
        return list(self.hdr[name]["shape"])

    def _read(self, off, nbytes):
        buf = bytearray(nbytes)
        with open(self.path, "rb", buffering=0) as f:
            f.seek(self.base + off)
            view, got = memoryview(buf), 0
            while got < nbytes:
                k = f.readinto(view[got:])
                if not k:
                    raise IOError(f"short read in {self.path}")
                got += k
        return buf

    def get(self, name):
        m = self.hdr[name]
        a, b = m["data_offsets"]
        dt = _ST_DTYPES[m["dtype"]]
        if b == a:
            return torch.empty(m["shape"], dtype=dt)
        return torch.frombuffer(self._read(a, b - a), dtype=dt).reshape(m["shape"])

    def get_rows(self, name, rows):
        """Rows along dim 0 (e.g. a few experts of a fused 3D tensor) without reading the rest."""
        m = self.hdr[name]
        a, b = m["data_offsets"]
        dt = _ST_DTYPES[m["dtype"]]
        per = (b - a) // m["shape"][0]
        out = [torch.frombuffer(self._read(a + int(r) * per, per), dtype=dt).reshape(m["shape"][1:]) for r in rows]
        return torch.stack(out)


class Checkpoint:
    """Read-only view of a (possibly sharded) safetensors checkpoint. Tensors are read lazily, one at a time."""

    def __init__(self, model_dir: str):
        self.dir = model_dir
        idx = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            fname = "model.safetensors"
            self.weight_map = {k: fname for k in _Shard(os.path.join(model_dir, fname)).keys()}
        with open(os.path.join(model_dir, "config.json")) as f:
            self.config = json.load(f)
        self.dims = Dims.from_config(self.config)
        self.arch = self.dims.arch
        if any(k.startswith("model.language_model.") for k in self.weight_map):
            self.prefix = "model.language_model."
        else:
            self.prefix = "model."
        self._rename = _RENAME.get(self.arch, {})
        self._unrename = {v: k for k, v in self._rename.items()}
        self._handles = {}

    def _h(self, fname):
        h = self._handles.get(fname)
        if h is None:
            h = self._handles[fname] = _Shard(os.path.join(self.dir, fname))   # header only: cheap to keep
        return h

    def full_name(self, l: int, short: str) -> str:
        """Checkpoint name of an internal layer-relative tensor name."""
        return f"{self.prefix}layers.{l}.{self._rename.get(short, short)}"

    def shared_mlp_names(self, l: int):
        return [self.full_name(l, n) for n in SHARED_MLP[self.arch]]

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        return self._h(self.weight_map[name]).get(name)

    def shape(self, name: str):
        return self._h(self.weight_map[name]).shape(name)

    def get_rows(self, name: str, rows) -> torch.Tensor:
        """Read only the given indices along dim 0 (e.g. a few experts of a fused 3D tensor)."""
        return self._h(self.weight_map[name]).get_rows(name, rows)

    def layer_names(self, l: int):
        p = f"{self.prefix}layers.{l}."
        return sorted(k for k in self.weight_map if k.startswith(p))

    def layer(self, l: int, skip_experts: bool = False) -> dict:
        """All tensors of decoder layer l, keyed by internal layer-relative names."""
        p = f"{self.prefix}layers.{l}."
        out = {}
        for k in self.layer_names(l):
            short = k[len(p):]
            short = self._unrename.get(short, short)
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

    def embed(self, ids: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
        """Input embeddings of token ids, scaled like the model does (Gemma: x sqrt(hidden) in bf16)."""
        e = self.get(self.embed_name())[ids].to(dtype)
        if self.dims.embed_scale != 1.0:
            e = e * torch.tensor(self.dims.embed_scale, dtype=dtype)
        return e

    def close(self):
        self._handles.clear()
