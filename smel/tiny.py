"""A small random Qwen3.5/3.6-MoE (same transformers class, reduced config) and a checkpoint in the real layout."""
import json
import os

import torch
from safetensors.torch import save_file


def tiny_text_config(n_layers=8, hidden=256, n_experts=32, top_k=4, vocab=512):
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    layer_types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(n_layers)]
    return Qwen3_5MoeTextConfig(
        vocab_size=vocab, hidden_size=hidden, num_hidden_layers=n_layers, layer_types=layer_types,
        full_attention_interval=4, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, num_experts=n_experts, num_experts_per_tok=top_k, moe_intermediate_size=64,
        shared_expert_intermediate_size=64, rms_norm_eps=1e-6, max_position_embeddings=4096,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [3, 3, 2], "mrope_interleaved": True},
        tie_word_embeddings=False, attention_bias=False, hidden_act="silu", initializer_range=0.02,
        output_router_logits=False,
    )


def build_tiny_model(seed=0, dtype=torch.float32, device="cuda", **kw):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
    torch.manual_seed(seed)
    cfg = tiny_text_config(**kw)
    cfg._attn_implementation = "sdpa"
    model = Qwen3_5MoeForCausalLM(cfg)
    with torch.no_grad():
        for n, p in model.named_parameters():
            # make norms and router non-trivial so the tests exercise (1 + w) and real routing
            if "linear_attn.norm" in n:
                p.normal_(1.0, 0.1)  # gated norm multiplies by w directly
            elif n.endswith("norm.weight"):
                p.normal_(0.0, 0.1)  # (1 + w) norms
            if n.endswith("mlp.gate.weight"):
                p.normal_(0.0, 0.3)
            if "experts.gate_up_proj" in n or "experts.down_proj" in n:
                p.normal_(0.0, 0.05)
    return model.to(device=device, dtype=dtype).eval()


def save_tiny_checkpoint(model, out_dir, n_shards=3):
    """Save in the layout of the real VL checkpoint: model.language_model.* + lm_head, sharded with an index."""
    os.makedirs(out_dir, exist_ok=True)
    sd = {}
    for k, v in model.state_dict().items():
        name = "model.language_model." + k[len("model."):] if k.startswith("model.") else k
        sd[name] = v.detach().to("cpu", torch.bfloat16).contiguous()
    keys = sorted(sd)
    per = (len(keys) + n_shards - 1) // n_shards
    wm = {}
    for s in range(n_shards):
        fname = f"model-{s + 1:05d}-of-{n_shards:05d}.safetensors"
        part = {k: sd[k] for k in keys[s * per:(s + 1) * per]}
        save_file(part, os.path.join(out_dir, fname), metadata={"format": "pt"})
        wm.update({k: fname for k in part})
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": wm}, f)
    cfg = model.config.to_dict()
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"model_type": "qwen3_5_moe", "text_config": cfg}, f, default=str)
    return out_dir
