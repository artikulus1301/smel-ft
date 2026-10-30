# Mixed-precision experts: running the fine-tuned Qwen3.6-35B-A3B in less RAM

The fine-tuned model of the [Qwen experiment](results_qwen.md) (L=20, Q3_K_XL GGUF, 16.0 GiB) runs in a desktop
companion app on a laptop with 24 GB RAM and an 8 GB GPU that also holds a TTS model. All experts live in RAM
(`--n-cpu-moe 40`), so llama-server takes ~15 GB. This experiment keeps the experts the app actually uses at the
original quantization and stores the rest in fewer bits. The router stays untouched: a rarely used expert is still
selected when needed, it is just computed from a coarser copy.

Hardware and software: RTX 4070 Laptop 8 GB, Ryzen 7 7435HS (8 cores, AVX2), 24 GB RAM, Windows 11; llama.cpp b11827
(970b51ce7) with [the patch](../inference/README.md).

## 1. How concentrated is the routing?

Workload: 10 dialogs of 10 exchanges on 10 new topics (not from the training data), a strong LLM playing the user of
the app, the app's own system prompts. Routing counts from `llama-imatrix`; hot sets ranked on 8 dialogs, checked on the
other 2 and on general Russian text ([scripts/expert_usage.py](../scripts/expert_usage.py)).

| hot experts per layer | share of experts | coverage, held-out topics | coverage, general text |
|---|---|---|---|
| 64 | 25% | 82.0% | 52.2% |
| 96 | 38% | 90.1% | 65.2% |
| 128 | 50% | 94.5% | 75.2% |
| 192 | 75% | 98.6% | 89.7% |

The hot set chosen on 8 topics covers the 2 unseen topics as well as its own (82.0% vs 81.8% at N=64), so it describes
the app, not the topics. Layers 0-2 are much flatter (57-59% at N=64 vs 76-97% elsewhere) and get 192 hot experts.

## 2. Variants

128 hot experts per layer (192 in layers 0-2); cold experts re-quantized from the merged bf16 weights with the importance
matrix of the source quant ([scripts/mixed_experts_gguf.py](../scripts/mixed_experts_gguf.py)). The source quant keeps
gate/up in IQ3_XXS and down in IQ4_XS.

| | cold gate/up | cold down | GGUF | KL vs original, held-out dialogs / general | same top-1 token |
|---|---|---|---|---|---|
| original | - | - | 16.04 GiB | - | - |
| v2 | IQ2_XXS | IQ2_XXS | 13.38 GiB | 0.030 / 0.017 | 93.6% / 95.5% |
| **v3** | IQ2_XXS | IQ3_XXS | **13.98 GiB** | **0.024** / 0.014 | 93.7% / 94.9% |
| v4 | IQ2_XXS | original bytes | 14.80 GiB | 0.031 / 0.014 | 93.9% / 95.8% |

`llama-perplexity --kl-divergence` against the original GGUF, 4 x 512 tokens per text.

## 3. Behaviour in the app

KL does not show everything. The app's own check (16-turn work conversation with the user's task list in the prompt,
ending with "summarize what I did today"; the user collected literature and did not start the draft):

| | summary claims the draft is written | the user addressed in the feminine |
|---|---|---|
| original (4 runs) | 0/4 | 0/4 |
| v2 (4 runs) | **3/4** | 2/4 |
| v3 (4 runs) | 0/4 | 3/4 |
| v4 (4 runs) | 0/4 | 2/4 |

v2 invents task facts; the coarser cold `down` projection is the likely cause - v3 and v4, which keep it at 3-4 bits,
do not. Persona metrics (breaking character, stage directions, empty answers) were 0 for every variant.

Blind pairwise judge (Mistral Large through an OpenAI-compatible API), **identical contexts**: every turn of the 10
reference dialogs of the original model, the candidate answers the same system prompt and history, both orders, a
verdict counts only if both orders agree ([examples/persona_eval/same_context_judge.py](../examples/persona_eval/same_context_judge.py)):

| | original wins | candidate wins | tie / orders disagree | sign test |
|---|---|---|---|---|
| v3 | 31 | 34 | 35 | p = 0.80 |
| v4 | 40 | 32 | 28 | p = 0.41 |

The judge cannot tell either variant from the original. (A first version of the judge compared answers from two
separately generated dialogs; it gave 89:8 for the original - an artifact: the dialogs diverge from the first turn,
and a judge shown one dialog's history prefers that dialog's answers.)

## 4. Speed and memory

llama-server with the app's flags (`-ngl 99 --n-cpu-moe 40 -ub 1024`, MTP speculative decoding), a planner prompt of
~2000 tokens plus a chat, 5 measured rounds:

| | generation | planner prompt | RAM of llama-server under load |
|---|---|---|---|
| original, stock llama.cpp | 27.9 tok/s (25.2-31.1) | 443 tok/s | 15.3 GB |
| v2, patched llama.cpp | 26.7 tok/s (25.1-28.3) | 456 tok/s | 12.75 GB |

v3 was measured in the app check only (answer median 4.4 s vs 5.4 s for the original, first token 2.4 s vs 3.6 s;
4 runs each, noisy); its RAM scales with the file, ~0.6 GB above v2. Right after loading, the patched build holds
1.3 GB instead of 3.7 GB: weights copied to the GPU leave the working set (Windows part of the patch).

The prompt is faster because llama.cpp copies the used experts to the GPU for every large batch and the cold experts
are smaller. Generation is unchanged once idle slots are skipped: in the first version of the patch the cold set
computed ~7.5 of the 8 slots of every token on a zero IQ2_XXS expert, and decoding fell to 17.7 tok/s - IQ2 dot products
are expensive on a CPU even when the weights are zero.

## 5. What did not work

**Letting Windows stream the experts** (hard working-set cap on llama-server, the rest of the mmap'd file in the
standby list): 29.4 tok/s at 12 GB, 24-25 at 9 GB, 20.5 at 7 GB, 15 at 5 GB; after another process took the RAM, a
2000-token prompt ran at 31 tok/s. The cost is page faults (millions of 4 KB pages), not the disk: the disk read only
0.1-0.4 GB per round. Reading one whole expert (1.3 MB) takes 0.15-0.27 ms from the file cache and 1.45 ms from this
laptop's NVMe, so even a custom expert cache could not keep 25 tok/s at 5-7 GB once the cache is evicted.

**Longer MTP drafts** (`--spec-draft-n-max` 3 or 4): 18.7 and 21.4 tok/s vs 29.4 with 2.

## 6. Correctness of the patch

A tiny random Qwen3.5-MoE (8 layers, 32 experts, top-4, logits scaled up so that any change shows), split into random
hot/cold sets with the cold bytes copied: KL vs stock llama.cpp ~0 and the same top-1 token at 100% on every path -
CPU, GPU, experts on the host, single-token and 16-token micro-batches, prompt batches offloaded to the GPU (with
`GGML_CUDA_DISABLE_FUSION=1` on both sides; with fusion the stock graph uses fused routing kernels the split graph
does not match, ~1e-5 relative difference). Two different hot sets give bit-identical outputs.

## Caveats

- One model (Qwen3.6-35B-A3B, `qwen35moe`), one machine, one workload; the hot set is specific to the app's prompts.
- 4 runs per variant in the app check: 3/4 vs 0/4 is a strong signal, 2/4 vs 3/4 is not.
- The judge is an LLM; it decided 65-72 of 100 pairs.
