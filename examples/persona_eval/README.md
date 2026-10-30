# Persona evaluation (reference experiments)

Scripts used in both reference experiments ([Qwen](../../docs/results_qwen.md), [Gemma 4](../../docs/results_gemma4.md)). They are tailored to a Russian-speaking
persona: the general prompts in `gen_compare.py`, and the persona text and regexes in `judge.py`, are examples to
adapt to your own domain.

| script | what it does |
|---|---|
| `run_ablation.py` | trains and evaluates the control variants one after another (skips finished steps) |
| `ablation_summary.py` | perplexity table of all `artifacts/adapter_*/eval_ppl.json` |
| `gen_compare.py` | starts `llama-server` with a GGUF, answers the held-out domain prompts + a few general ones (same seed per prompt), `--report` puts several runs side by side |
| `judge.py` | automatic metrics (persona breaks, stage directions, length) and a blind pairwise judge: the base model compares two anonymous answers to the dataset reference, in both orders |
| `same_context_judge.py` | a candidate GGUF answers every turn of reference dialogs (same system prompt and history) and an LLM behind an OpenAI-compatible API judges it against the recorded answer, both orders; used for the [mixed-precision experts](../../docs/results_mixed_experts.md) |

`llama-server` is found through env `SMEL_LLAMA_SERVER`, `llamacpp.dir` in the config, or `PATH`. The default flags
offload the experts of 32 layers to the CPU and use Qwen's MTP head, which suits an 8 GB GPU; replace them with
`SMEL_LLAMA_SERVER_FLAGS` (e.g. Gemma 4: `-ngl 99 --n-cpu-moe 20 -c 8192 -fa on --no-mmproj --jinja`) or append with
`SMEL_LLAMA_SERVER_ARGS`.

```bash
python examples/persona_eval/gen_compare.py --gguf base.gguf --tag before --n-domain 50
python scripts/export_gguf.py --adapter artifacts/adapter_L20 --base-gguf base.gguf --imatrix imatrix.gguf --out work/gguf/L20.gguf
python examples/persona_eval/gen_compare.py --gguf work/gguf/L20.gguf --tag L20 --n-domain 50
python examples/persona_eval/judge.py --gguf base.gguf --tags before L20 --pairs L20:before

# two models on identical contexts (dialogs.jsonl: dialogs of the reference model, one per line)
export SMEL_JUDGE_API_KEY=...  SMEL_JUDGE_BASE_URL=https://api.example.com/v1  SMEL_JUDGE_MODEL=judge-model
python examples/persona_eval/same_context_judge.py dialogs.jsonl --gguf mixed.gguf --names original mixed --out work/judge/mixed.md
```

Compare answers to the same context, not two dialogs generated separately: they diverge from the first turn, and a
judge shown one dialog's history prefers that dialog's answers (in the reference run this gave a spurious 89 : 8).
