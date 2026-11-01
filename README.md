# SMEL FT — Sparse MoE LoRA Fine-Tuning

[Русская версия](README.ru.md)

Fine-tune a 26-35B Mixture-of-Experts model on **one 8 GB consumer GPU**: LoRA only on the experts that the router
actually uses for your domain, with the frozen bottom of the network computed once and cached.

Tested end to end on two different MoE families, on an RTX 4070 Laptop (8 GB) with 24 GB RAM under Windows 11:

| model | trained block | 3 epochs | peak VRAM | held-out domain perplexity | persona breaks | details |
|---|---|---|---|---|---|---|
| **Qwen3.6-35B-A3B** | layers 16-39 / 20-39 | 18-27 min | 4.6-4.8 GB | 17.9 -> 8.0-8.3 | 20% -> 0-2% | [docs/results_qwen.md](docs/results_qwen.md) |
| **Gemma 4 26B A4B** | layers 12-29 | 21-22 min | 5.5 GB | 128 -> 8.6 | 20% -> 6-8% | [docs/results_gemma4.md](docs/results_gemma4.md) |

Both runs end with a GGUF of the same size and speed as the original quant, and neither shows forgetting on
general text. The task in both was a persona fine-tune on 1000 Russian chat dialogs (61k supervised tokens).

## Why

- In transformers 5.x and in the checkpoints, the experts of Qwen3.5/3.6-MoE and Gemma 4 are **fused 3D tensors**
  (e.g. `gate_up_proj [256, 1024, 2048]` per Qwen layer). PEFT cannot put LoRA on a single expert, and
  bitsandbytes does not quantize these tensors.
- 4-bit QLoRA is not recommended for these families, and LoRA through the usual tools needs the whole model on the
  GPU (Unsloth lists > 40 GB of VRAM for Gemma 4 26B A4B).
- In a MoE layer, a domain mostly uses a small subset of experts, so training only those is cheap.

## How it works

```
probe    run all layers once, layer by layer, over domain and general text; record the router distribution
select   D_l = KL(p_domain || p_general) per layer -> cut layer L; per layer the smallest expert set that covers
         p = 50% of the domain gate mass (<= k_max = 12); budget estimate for VRAM / RAM / time
cache    hidden states entering layer L for the whole dataset (a free by-product of probe)
pack     block L..N-1: frozen experts -> int4 (group 64), selected experts and dense parts -> bf16, lm_head -> int8
train    LoRA on the selected experts + the always-active MLP of every block layer (Qwen: shared expert,
         Gemma 4: dense MLP), router frozen; layer-major schedule: each layer is loaded to the GPU twice per
         optimizer step (forward for all micro-batches, then backward with recompute)
merge    LoRA merged exactly into the bf16 weights (unchanged shards are hard-linked)
export   GGUF: a copy of an existing quant of the base model in which only the trained tensors are re-quantized,
         with the same ggml type and importance matrix; all other bytes stay identical
```

The rest of the implementation:
- **Own MoE forward.** Frozen experts run in a custom autograd function that dequantizes weights again, group by
  group, in backward, so a layer of experts is never held for autograd. Per-expert matmuls use `torch._grouped_mm`;
  int4 dequantization is a Triton kernel. The combine step is deterministic (no atomics).
- **Layers.** Qwen: Gated DeltaNet on flash-linear-attention kernels and gated full attention. Gemma 4:
  sliding-window and global attention (K = V, partial proportional RoPE, head_dim 512 via memory-efficient SDPA),
  dense MLP in parallel with the experts. Sequence packing works through `cu_seqlens`, a boundary-aware causal conv
  and block-diagonal / windowed masks, all checked by tests.
- **Loss.** Chunked cross-entropy over the 248-262k vocabulary without materializing logits, with Gemma's logit
  softcapping. One pass over the head weight: the gradient is accumulated online with the same running maximum as
  the log-sum-exp (35% less time than two passes at the Qwen vocabulary). Only the assistant answers are supervised; the training text matches the generation prompt with
  thinking off byte for byte.
- **Checkpoint access.** Tensors (or single expert rows) are read with plain file reads, never by mapping a whole
  shard: on Windows a 50 GB shard does not fit the commit limit.
- **Fused kernels.** The norms of the frozen layers are Triton kernels, one launch each way: RMSNorm, the residual
  add fused with the norm after it, and the gated output norm of Gated DeltaNet (backward gives dx only, the norm
  weights are frozen; fp32 and CPU keep the torch code, which the exact gradient tests use). The LoRA add is the
  epilogue of its GEMM (`addmm`), and gate / up of an MLP share one base GEMM and one LoRA-A GEMM. On one Qwen3.6
  layer at full size (`scripts/bench/bench_layer.py`, 4092 tokens): forward −9…−10%, recompute + backward −8…−10%,
  activation peak up to −13% (Gated DeltaNet layers).
- **Weight streaming.** Double-buffered prefetch on a side CUDA stream, from pinned RAM or from disk.
- **Resumable stages.** Long stages can be continued with `--resume`, and every stage appends a summary to
  `logs/progress.md`.

## Supported models

Architectures of transformers; sizes are read from `config.json`.

| architecture | tested model | trained next to the selected experts | config |
|---|---|---|---|
| `qwen3_5_moe` (Qwen3.5/3.6 MoE) | `Qwen/Qwen3.6-35B-A3B` | shared expert | [configs/default.yaml](configs/default.yaml) |
| `gemma4` with a MoE block | `google/gemma-4-26B-A4B-it` | dense MLP | [configs/gemma4.yaml](configs/gemma4.yaml) |

Both paths are checked against transformers on tiny random models of the same class (`tests/`), and the Gemma 4
layers additionally on the real weights.

## Requirements

- An NVIDIA GPU with ~6 GB free VRAM (Ampere or newer: bf16 and Triton). Tested on an RTX 4070 Laptop, 8 GB.
- 24 GB RAM tested (the RAM budget is a config value; less was not tried).
- Disk: the bf16 weights (Qwen 72 GB, Gemma 52 GB) + ~15-25 GB for the cache and the packed block + the GGUF.
- Python 3.11+ (tested on 3.14), torch 2.11+ with CUDA, transformers 5.17+, Triton
  (`triton-windows` on Windows), flash-linear-attention (Qwen only). Linux and native Windows work; WSL is not needed.
- For GGUF export and evaluation: a [llama.cpp](https://github.com/ggml-org/llama.cpp) build (`llama-server`,
  `ggml-base`), plus an existing GGUF of the base model and its importance matrix.

```bash
pip install -e ".[gguf,data,dev]"      # install torch with CUDA first, from pytorch.org
python -m pytest                       # ~40 s; needs CUDA; downloads only the tokenizer files
```

## Quick start

```bash
# 0. data: your chat jsonl -> data/domain.jsonl, one dialog per line:
#    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
python scripts/download.py                    # Qwen; for Gemma 4: --repo google/gemma-4-26B-A4B-it
python scripts/make_general.py                # reference text -> data/general.jsonl (Russian wiki + chats);
                                              # or put your own general text there
# for Gemma 4 add --config configs/gemma4.yaml to every command below

# optional dry run of the whole pipeline on a tiny random model of the same class (minutes):
python scripts/make_tiny_e2e.py --general data/general.jsonl          # --arch gemma4 for Gemma
#   ... then every stage below with --config configs/tiny_e2e.yaml (configs/tiny_e2e_gemma4.yaml)

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

All settings live in the config and can be overridden per run, e.g. `--set train.lr=2e-4 train.epochs=4`.

## Stages and outputs

| stage | script | output |
|---|---|---|
| download | `scripts/download.py [--repo R]` | `paths.model_dir` |
| general corpus | `scripts/make_general.py` | `paths.general_data` |
| probe | `scripts/probe.py` | `artifacts/routing_stats.pt`, `reports/probe.md`, `reports/probe_divergence.png`, cached inputs of the candidate L |
| select | `scripts/select_experts.py` | `paths.selection`, `reports/select.md` |
| cache | `scripts/cache.py [--L N]` | `work/cache_L{L}/` |
| pack | `scripts/pack.py [--no-write]` | `work/packed_L{L}/`, `reports/pack.md` |
| train | `scripts/train.py [--max-steps N] [--resume] [--tag T] [--selection F]` | `artifacts/adapter_{T}/`, `logs/train_{T}.jsonl` |
| eval | `scripts/eval.py --adapter A [--packed]` | `reports/eval.md`, `A/eval_ppl.json` |
| merge | `scripts/merge.py --adapter A --out DIR [--verify]` | merged safetensors |
| GGUF | `scripts/export_gguf.py --adapter A --base-gguf G [--imatrix I] [--reuse]` | patched GGUF |
| ablations | `scripts/make_selection.py --kind random\|shared` | control expert selections |

Tools: `scripts/gguf_repro_check.py` checks that the quantization path reproduces the bytes of the base GGUF on an
unchanged layer. `scripts/gguf_quant_error.py` and `scripts/bench/` hold hardware and kernel benchmarks;
`bench_layer.py --trace P` writes a torch.profiler trace of one layer step, and `scripts/bench/trace_gaps.py` shows
where the GPU waits for the host in it. `tests/test_recompile.py` checks that no Triton kernel compiles again for a
new micro-batch length.
[examples/persona_eval/](examples/persona_eval/) is the generation + blind-judge evaluation used in both experiments
(llama-server flags via `SMEL_LLAMA_SERVER_FLAGS`).

## What the experiments showed

**The probe finds the same structure in both families.** Domain-specific routing sits in the middle layers
(Qwen 10-28 of 40, Gemma 12-22 of 30), while the top layers route domain and general text to the same experts.
The 70% coverage rule gives the same cut layer for both, L = 12-13.

**Qwen: what the router-based choice adds.** Variants differ only in which routed experts get LoRA; the shared
experts are trained in all of them (perplexity before: 17.88):

| variant | domain perplexity | judge vs base model |
|---|---|---|
| shared experts only | 9.03 | 19 : 20 |
| + 168 random experts (L=20) | 8.88 | 23 : 15 |
| + 168 router-selected experts (L=20) | 8.34 | **27 : 10** |
| + 213 router-selected experts (L=16) | **7.97** | 24 : 19 |

The shared experts carry most of the effect. Router-selected experts add a consistent extra 8-12% on perplexity
(random ones add 2%) and most of the persona voice. A lower cut layer helps, as the probe suggests.

**Gemma 4: the method transfers, the persona is harder.** Persona breaks fell from 20% to 6% and stage directions
from 54% to 2%, but the judge's preference over the base model (18 : 10) is not significant on 50 prompts. All
remaining breaks are in dialogs without a system prompt, where Gemma falls back to its own identity; routing drift
during training was larger than in Qwen (up to 35% in three layers).

These are honest small-scale numbers from one domain; see the two result documents for the caveats.

## Running the fine-tuned model in less RAM

The same router profiling applies after training. Most expert selections of an application go to a small part of the
experts, so [inference/](inference/README.md) keeps the most used experts of every layer at the original quantization
and re-quantizes the rest to fewer bits, with a small llama.cpp patch to run the result. The router is unchanged: a
rarely used expert is still selected, just computed from a coarser copy. On the fine-tuned Qwen in a desktop companion
app ([docs/results_mixed_experts.md](docs/results_mixed_experts.md)):

| | GGUF | RAM of llama-server | generation | blind judge vs original |
|---|---|---|---|---|
| original Q3_K_XL | 16.0 GiB | 15.3 GB | 27.9 tok/s | - |
| hot 128/256 experts per layer, cold gate/up IQ2_XXS, cold down IQ3_XXS | 14.0 GiB | ~13.3 GB* | within noise* | 31 : 34, p = 0.80 |

\* measured on the variant with cold `down` in IQ2_XXS too (12.75 GB, 26.7 vs 27.9 tok/s over 5 rounds); this one is
0.6 GiB larger.

KL alone was not enough to pick the variant: with the cold `down` projection in IQ2_XXS (KL 0.030) the model started
to misreport the user's tasks in the app's own check; with IQ3_XXS it did not.

## Limitations

- The adapter learns the training answers closely: their length, formatting and defects included. If the data is
  worse than what the base model already writes (short machine-translated answers, broken punctuation), the
  fine-tuned model gets worse in conversation even while the domain perplexity drops. Check the data first, and
  compare with the base model by a judge or a benchmark, not by perplexity alone.
- Two architectures (`qwen3_5_moe`, `gemma4` with MoE). Other MoE families need their own layer code in
  `smel/layer_runner.py` and names in `smel/model_io.py`.
- The router stays frozen, and LoRA has no dropout. The experts are chosen once, on the model before training.
- With a block larger than the OS file cache, each step re-reads the packed weights from disk. Fast NVMe matters.
- GGUF export patches an existing quant; it does not create one from scratch.
- `scripts/merge.py` loads every shard it rewrites into RAM, so merging into safetensors does not work for shards
  larger than RAM (Gemma 4's 50 GB shard). GGUF export does not need it.
- Mixed-precision experts: the llama.cpp patch covers `qwen35moe` only and is made against one llama.cpp build
  (b11827); the hot set is as good as the text it was ranked on.

## Project layout

```
smel/       library: model_io (checkpoint access, model dimensions), moe, quant (int4 + Triton dequant), lora,
            layer_runner (Qwen and Gemma 4 layers, layer-major runner), norm (Triton norms), stream (prefetch),
            loss (chunked CE),
            data (chat templates + masks + packing), bottom (layer-by-layer pass of the frozen layers),
            pack (packing + exact merge), gguf_patch, pipeline (config, stores), tiny (test models),
            mixed_experts (hot/cold expert GGUF)
scripts/    pipeline stages, expert_usage + mixed_experts_gguf (inference), scripts/bench/ benchmarks
configs/    default.yaml (Qwen), gemma4.yaml
tests/      equivalence tests against transformers on tiny random models of both classes, mixed-experts GGUF layout
examples/   persona_eval: generation, automatic metrics, blind pairwise judges (reference answers; identical contexts)
inference/  mixed-precision experts: llama.cpp patch and how-to
docs/       results of the reference experiments
```

## License

[Apache-2.0](LICENSE). The llama.cpp patch in `inference/llama_cpp_patches/` modifies llama.cpp and is under its MIT
license. The base models, their weights and tokenizers are not included and keep their own licenses. See [NOTICE](NOTICE).
