# SMEL FT — Sparse MoE LoRA Fine-Tuning

[Русская версия](README.ru.md)

Fine-tune a 35B Mixture-of-Experts model on **one 8 GB consumer GPU**: LoRA only on the experts that the router
actually uses for your domain, with the frozen bottom of the network computed once and cached.

Reference run (details in [docs/results.md](docs/results.md)): **Qwen3.6-35B-A3B** on an RTX 4070 Laptop (8 GB) with
24 GB RAM, Windows 11. 3 epochs took 18-27 minutes at under 5 GB peak VRAM. Held-out domain perplexity dropped from
17.88 to 7.97-8.34, with no measurable forgetting on general text. The result is a GGUF of the same size and speed as
the original quant.

## Why

- In transformers 5.x and in the checkpoint, the experts of Qwen3.5/3.6-MoE are **fused 3D tensors**
  (`gate_up_proj [256, 1024, 2048]`, `down_proj [256, 2048, 512]` per layer). PEFT cannot put LoRA on a single
  expert, and bitsandbytes does not quantize these tensors.
- 4-bit QLoRA is not recommended for this model family, and it would still need the whole 35B model on the GPU.
- In a MoE layer, a domain mostly uses a small subset of experts, so training only those is cheap.

## How it works

```
probe    run all 40 layers once, layer by layer, over domain and general text; record the router distribution
select   D_l = KL(p_domain || p_general) per layer -> cut layer L; per layer the smallest expert set that covers
         p = 50% of the domain gate mass (<= k_max = 12); budget estimate for VRAM / RAM / time
cache    hidden states entering layer L for the whole dataset (a free by-product of probe)
pack     block L..N-1: frozen experts -> int4 (group 64), selected experts and dense parts -> bf16, lm_head -> int8
train    LoRA on the selected experts + the shared expert of every block layer, router frozen;
         layer-major schedule: each layer is loaded to the GPU twice per optimizer step
         (forward for all micro-batches, then backward with recompute), whatever the number of micro-batches
merge    LoRA merged exactly into the bf16 weights (unchanged shards are hard-linked)
export   GGUF: a copy of an existing quant of the base model in which only the trained experts are re-quantized,
         with the same ggml type and importance matrix; all other bytes stay identical
```

The rest of the implementation:
- **Own MoE forward.** Frozen experts run in a custom autograd function that dequantizes weights again, group by
  group, in backward, so a layer of experts is never held for autograd. Per-expert matmuls use `torch._grouped_mm`;
  int4 dequantization is a Triton kernel. The combine step is deterministic (no atomics).
- **Hybrid layers.** Gated DeltaNet runs on flash-linear-attention kernels; full attention uses SDPA. Sequence
  packing works through `cu_seqlens`, a boundary-aware causal conv and a block-diagonal mask, all checked by tests.
- **Loss.** Chunked cross-entropy over the 248k vocabulary, without materializing logits. Only the assistant
  answers are supervised; the empty `<think></think>` block of the template is excluded.
- **Weight streaming.** Double-buffered prefetch on a side CUDA stream, from pinned RAM or from mmap.
- **Resumable stages.** Long stages can be continued with `--resume`, and every stage appends a summary to
  `logs/progress.md`.

**Supported models** (transformers architectures; sizes are read from `config.json`):

| architecture | tested model | trained next to the selected experts | config |
|---|---|---|---|
| `qwen3_5_moe` (Qwen3.5/3.6 MoE) | `Qwen/Qwen3.6-35B-A3B` (reference run below) | shared expert | `configs/default.yaml` |
| `gemma4` with a MoE block | `google/gemma-4-26B-A4B-it` | dense MLP (runs in parallel with the experts) | `configs/gemma4.yaml` |

Gemma 4 specifics handled: sliding-window (1024) and global attention (K = V, partial proportional RoPE,
head_dim 512 via memory-efficient SDPA), the router's own norm and per-expert scale, GELU-tanh experts, tied
embeddings, final logit softcapping, and its chat template (an empty thought channel is added to answers exactly as
llama-server does with thinking off). The Gemma 4 path is checked against transformers by `tests/test_gemma4.py`;
a full run on the real model is in progress.

## Requirements

- An NVIDIA GPU with ~6 GB free VRAM (Ampere or newer: bf16 and Triton). Tested on an RTX 4070 Laptop, 8 GB.
- 24 GB RAM tested (the RAM budget is a config value; less was not tried); ~110 GB of disk for the bf16 weights (72 GB), the cache and the packed block.
- Python 3.11+ (tested on 3.14), torch 2.11+ with CUDA, transformers 5.17+, Triton
  (`triton-windows` on Windows), flash-linear-attention. Linux and native Windows work; WSL is not needed.
- For GGUF export and evaluation: a [llama.cpp](https://github.com/ggml-org/llama.cpp) build (`llama-server`,
  `ggml-base`), plus an existing GGUF of the base model and its importance matrix.

```bash
pip install -e ".[gguf,data,dev]"      # install torch with CUDA first, from pytorch.org
python -m pytest                       # ~30 s; needs CUDA; downloads only the Qwen tokenizer files
```

## Quick start

```bash
# 0. data: your chat jsonl -> data/domain.jsonl, one dialog per line:
#    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
python scripts/download.py                    # bf16 safetensors -> data/Qwen3.6-35B-A3B (72 GB, resumable)
python scripts/make_general.py                # reference text -> data/general.jsonl (Russian wiki + chats);
                                              # or put your own general text there

# optional dry run of the whole pipeline on a tiny random model of the same class (minutes):
python scripts/make_tiny_e2e.py --general data/general.jsonl
#   ... then every stage below with --config configs/tiny_e2e.yaml

python scripts/probe.py                       # router profile, ~6 min; reports/probe.md
python scripts/select_experts.py              # L + experts + memory/time estimate; configs/selection.yaml
#   to force a cut layer: --set select.L_override=16
python scripts/cache.py                       # input of layer L (hard-linked from the probe if available)
python scripts/pack.py                        # packed block + check against bf16 on 50 dialogs
python scripts/train.py --max-steps 5         # smoke test, then:
python scripts/train.py --resume --set train.lr=2e-4
python scripts/eval.py --adapter artifacts/adapter_L16
python scripts/merge.py --adapter artifacts/adapter_L16 --out work/merged_L16 --verify
SMEL_LLAMA_CPP_DIR=/path/to/llama.cpp/build/bin \
python scripts/export_gguf.py --adapter artifacts/adapter_L16 --base-gguf base.gguf --imatrix imatrix.gguf
```

All settings live in [configs/default.yaml](configs/default.yaml) and can be overridden per run, e.g.
`--set train.lr=2e-4 train.epochs=4`.

## Stages and outputs

| stage | script | output |
|---|---|---|
| download | `scripts/download.py` | `paths.model_dir` |
| general corpus | `scripts/make_general.py` | `paths.general_data` |
| probe | `scripts/probe.py` | `artifacts/routing_stats.pt`, `reports/probe.md`, `reports/probe_divergence.png`, cached inputs of the candidate L |
| select | `scripts/select_experts.py` | `configs/selection.yaml`, `reports/select.md` |
| cache | `scripts/cache.py [--L N]` | `work/cache_L{L}/` |
| pack | `scripts/pack.py [--no-write]` | `work/packed_L{L}/`, `reports/pack.md` |
| train | `scripts/train.py [--max-steps N] [--resume] [--tag T] [--selection F]` | `artifacts/adapter_{T}/`, `logs/train_{T}.jsonl` |
| eval | `scripts/eval.py --adapter A [--packed]` | `reports/eval.md`, `A/eval_ppl.json` |
| merge | `scripts/merge.py --adapter A --out DIR [--verify]` | merged safetensors |
| GGUF | `scripts/export_gguf.py --adapter A --base-gguf G [--imatrix I] [--reuse]` | patched GGUF |
| ablations | `scripts/make_selection.py --kind random\|shared` | control expert selections |

Tools: `scripts/gguf_repro_check.py` checks that the quantization path reproduces the bytes of the base GGUF on an
unchanged layer. `scripts/gguf_quant_error.py` and `scripts/bench/` hold hardware and kernel benchmarks.
[examples/persona_eval/](examples/persona_eval/) is the generation + blind-judge evaluation of the reference experiment.

## What the reference experiment showed

Variants differ only in which routed experts get LoRA; the shared experts are trained in all of them
(perplexity before: 17.88):

| variant | domain perplexity | judge vs base model |
|---|---|---|
| shared experts only | 9.03 | 19 : 20 |
| + 168 random experts (L=20) | 8.88 | 23 : 15 |
| + 168 router-selected experts (L=20) | 8.34 | **27 : 10** |
| + 213 router-selected experts (L=16) | **7.97** | 24 : 19 |

The shared experts carry most of the effect. Router-selected experts add a consistent extra 8-12% on perplexity
(random ones add 2%) and most of the persona voice. A lower cut layer helps, as the probe suggests.
These are honest small-scale numbers from one domain; see [docs/results.md](docs/results.md) for the caveats.

## Limitations

- Two architectures (`qwen3_5_moe`, `gemma4` with MoE). Other MoE families need their own layer code in
  `smel/layer_runner.py` and names in `smel/model_io.py`.

## Project layout

```
smel/       library: model_io (mmap layer loading), moe, quant (int4 + Triton dequant), lora, layer_runner,
            stream (prefetch), loss (chunked CE), data (chat template + masks + packing), bottom (layer-by-layer
            pass of the frozen layers), pack (packing + exact merge), gguf_patch, pipeline (config, stores)
scripts/    pipeline stages, scripts/bench/ benchmarks
tests/      equivalence tests against transformers on a tiny random model of the same class
examples/   persona_eval: generation, automatic metrics and a blind pairwise judge through llama-server
docs/       results of the reference experiment
```

## License

[Apache-2.0](LICENSE). The base model, its weights and tokenizer are not included and keep their own license.
See [NOTICE](NOTICE).
