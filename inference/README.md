# Inference: mixed-precision experts

A fine-tuned MoE spends most of its memory on experts that the application rarely selects. This directory and two
scripts make a GGUF where the most used experts of every layer keep the original quantization and the rest are stored
in fewer bits, plus a small llama.cpp patch that runs it. On the reference app (Qwen3.6-35B-A3B, see
[docs/results_mixed_experts.md](../docs/results_mixed_experts.md)) the file shrinks from 16.0 to 14.0 GiB and llama-server
needs ~2 GB less RAM at the same generation speed, and a blind judge cannot tell the answers from the original's.

Nothing is removed: the router is unchanged and a cold expert is still used when the router picks it.

## The patch

`llama_cpp_patches/mixed-experts-b11827.patch`, against llama.cpp **b11827** (commit 970b51ce7); MIT, like llama.cpp.

- `qwen35moe` loads `blk.L.ffn_{gate,up,down}_exps_cold` and three small tables per layer when the GGUF has them
  (other GGUFs load as before);
- `build_moe_ffn` runs the hot and the cold set and adds the results. Each slot of a token goes to its own set; the
  other set gets id -1 for it, which the CPU `mul_mat_id` skips (zero output row). Batches the CUDA backend computes
  (prompts offloaded to the GPU, experts on the GPU) need a valid, distinct expert per slot: there each set takes its
  own top-k by router probability, keeps the router's weights for the selected experts and 0 for the rest, and puts the
  outputs back in router order - the sum is the same as without the split;
- `ggml-cpu`: `mul_mat_id` and its repacked variant skip id -1;
- Windows: weights copied to the GPU at load time leave the process working set (`VirtualUnlock`), as `munmap` of the
  unused fragments already does on Linux. Useful for any model with experts on the CPU.

Applying and building:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 970b51ce7
git apply ../smel-ft/inference/llama_cpp_patches/mixed-experts-b11827.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target llama-server
```

On Windows without a CUDA build environment it is enough to rebuild `llama.dll` and the CPU backend you use
(`-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=OFF`, targets `llama` and e.g. `ggml-cpu-haswell`) and copy
them into a copy of the official release of the same build: the C API between them does not change. An MSVC build of
`ggml-cpu` measured as fast as the release's Clang one on a Ryzen 7 7435HS.

## Making a mixed GGUF

1. **Routing counts on your workload.** Text that looks like what the model will really do - for a chat app, dialogs
   with its real system prompts. Hold some topics out to check that the hot set carries over.

   ```bash
   python scripts/expert_usage.py render --dialogs dialogs.jsonl --tokenizer MODEL_DIR --holdout 5 10 \
       --general general.jsonl --out-dir work/usage
   llama-imatrix -m model.gguf -f work/usage/calib_train.txt -o work/usage/imatrix_train.gguf -c 512 --parse-special
   llama-imatrix -m model.gguf -f work/usage/calib_holdout.txt -o work/usage/imatrix_holdout.gguf -c 512 --parse-special
   python scripts/expert_usage.py report --out-dir work/usage      # coverage table: how many hot experts you need
   ```

2. **Build.** The hot experts are copied byte for byte; the cold ones are re-quantized from the merged model (the
   output of `scripts/merge.py`) with the importance matrix the source quant was made with.

   ```bash
   python scripts/mixed_experts_gguf.py --src model.gguf --out mixed.gguf \
       --usage work/usage/imatrix_train.gguf work/usage/imatrix_holdout.gguf \
       --hot 128 --hot-layer 0:192 1:192 2:192 --cold-type IQ2_XXS --cold-down IQ3_XXS \
       --hf merged_model_dir --imatrix imatrix_of_the_source_quant.gguf
   ```

   `--cold-type copy` keeps the cold bytes too: the result must then compute exactly what the source does (KL ~0 in
   `llama-perplexity --kl-divergence`), which tests a build of the patch.

3. **Check on the real task, not only KL.** In the reference experiment, cold `down` in IQ2_XXS had a low KL (0.030) but
   made the model misreport the user's tasks; `down` in IQ3_XXS did not.
   [examples/persona_eval/same_context_judge.py](../examples/persona_eval/same_context_judge.py) compares two models on
   identical contexts.

## Limitations

- Only the `qwen35moe` architecture (Qwen3.5 / 3.6 MoE) is wired in the patch; other MoE architectures need the same few
  lines in their `models/*.cpp` (separate gate/up experts only, no per-expert biases or scales).
- Tested on one build (b11827) and one machine; the patch will need rebasing for other llama.cpp versions.
- The MoE GPU cache (`--moe-cache-mib`) is not used for split layers.
- In the CUDA batched path each set computes all slots of a token (the inactive ones with weight 0), so the expert
  matmuls of a prompt cost twice the compute on the GPU. Prompts were still faster in the reference run, because fewer
  bytes are copied to the GPU.
