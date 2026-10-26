# Persona evaluation (reference experiment)

Scripts used for the ablation in [docs/results.md](../../docs/results.md). They are tailored to a Russian-speaking
persona: the general prompts in `gen_compare.py`, and the persona text and regexes in `judge.py`, are examples to
adapt to your own domain.

| script | what it does |
|---|---|
| `run_ablation.py` | trains and evaluates the control variants one after another (skips finished steps) |
| `ablation_summary.py` | perplexity table of all `artifacts/adapter_*/eval_ppl.json` |
| `gen_compare.py` | starts `llama-server` with a GGUF, answers the held-out domain prompts + a few general ones (same seed per prompt), `--report` puts several runs side by side |
| `judge.py` | automatic metrics (persona breaks, stage directions, length) and a blind pairwise judge: the base model compares two anonymous answers to the dataset reference, in both orders |

`llama-server` is found through env `SMEL_LLAMA_SERVER`, `llamacpp.dir` in the config, or `PATH`. The default flags
offload the experts of 32 layers to the CPU, which suits an 8 GB GPU; add or override flags with
`SMEL_LLAMA_SERVER_ARGS`.

```bash
python examples/persona_eval/gen_compare.py --gguf base.gguf --tag before --n-domain 50
python scripts/export_gguf.py --adapter artifacts/adapter_L20 --base-gguf base.gguf --imatrix imatrix.gguf --out work/gguf/L20.gguf
python examples/persona_eval/gen_compare.py --gguf work/gguf/L20.gguf --tag L20 --n-domain 50
python examples/persona_eval/judge.py --gguf base.gguf --tags before L20 --pairs L20:before
```
