"""Stage 2: router profiling on domain vs general text, layer by layer through the whole bf16 model.

Also saves the inputs of the candidate cut layers (cache for free). Output: artifacts/routing_stats.pt,
reports/probe.md (+ png), work_dir/probe/L{l}/h_{domain,general}.npy.
Usage: python scripts/probe.py [--config c.yaml] [--set probe.mb_tokens=4096] [--fresh]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smel.bottom import run_bottom  # noqa: E402
from smel.model_io import Checkpoint  # noqa: E402
from smel.pipeline import JsonlLog, append_progress, build_token_sets, load_config, parse_overrides  # noqa: E402


def js(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (p * (p / m).log()).sum(-1) + 0.5 * (q * (q / m).log()).sum(-1)


def write_report(cfg, stats, d, path_md, path_png):
    eps = 1e-12
    v = cfg["select"]["stats_view"]
    rows = []
    P, views, GM = {}, {}, {}
    for name in ("domain", "general"):
        s = stats[name]
        view = views[name] = v if s["ntok"][v].sum() > 0 else "all"  # plain text has no 'resp' tokens
        P[name] = (s["d"][view]["prob_sum"] / s["ntok"][view][:, None].clamp_min(1)).clamp_min(eps)
        P[name] = P[name] / P[name].sum(-1, keepdim=True)
        gm = s["d"][view]["gate_mass"]
        GM[name] = gm / gm.sum(-1, keepdim=True).clamp_min(eps)
    kl = (P["domain"] * (P["domain"] / P["general"]).log()).sum(-1)
    jsd = js(P["domain"], P["general"])
    gm_dom, gm_gen = GM["domain"], GM["general"]
    for l in range(d.n_layers):
        srt = gm_dom[l].sort(descending=True)
        k50 = int((srt.values.cumsum(0) < 0.5).sum()) + 1
        top = srt.indices[:5].tolist()
        enr = (gm_dom[l] / gm_gen[l].clamp_min(1e-6))
        rows.append(f"| {l} | {d.layer_types[l][:4]} | {kl[l]:.4f} | {jsd[l]:.4f} | {k50} | "
                    f"{', '.join(f'{e}({gm_dom[l, e]*100:.1f}%/×{enr[e]:.1f})' for e in top)} |")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].bar(range(d.n_layers), kl.numpy(), label="KL(dom||gen)")
        ax[0].plot(range(d.n_layers), jsd.numpy() * float(kl.max() / max(jsd.max(), 1e-9)), "r.-",
                   label="JS (rescaled)")
        ax[0].set_xlabel("layer")
        ax[0].legend()
        ax[0].set_title("routing divergence domain vs general")
        tail = kl.flip(0).cumsum(0).flip(0) / kl.sum()
        ax[1].plot(range(d.n_layers), tail.numpy(), "g.-")
        ax[1].axhline(cfg["select"]["coverage"], color="gray", ls="--")
        ax[1].set_xlabel("cut layer L")
        ax[1].set_ylabel("share of sum D_l in layers >= L")
        fig.tight_layout()
        fig.savefig(path_png, dpi=110)
    except Exception as e:  # pragma: no cover
        print("plot failed:", e)
    ntd = stats["domain"]["ntok"][views["domain"]][0]
    ntg = stats["general"]["ntok"][views["general"]][0]
    md = [f"# Probe: router profile\n",
          f"Tokens in the statistics: domain {int(ntd)} ({views['domain']}), general {int(ntg)} ({views['general']}).\n",
          f"![divergence]({os.path.basename(path_png)})\n",
          "D_l = KL(p_domain,l || p_general,l) of the mean router distribution (softmax over all experts). "
          "k50 = number of experts covering 50% of the domain gate mass. Top-5 experts: share of the domain gate mass "
          "and enrichment vs general text.\n",
          "| layer | type | KL | JS | k50 | top-5 domain experts |", "|---|---|---|---|---|---|"] + rows
    open(path_md, "w", encoding="utf-8").write("\n".join(md) + "\n")
    return kl, jsd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["paths"]["tokenizer"])
    sets = build_token_sets(cfg, tok)
    d = Checkpoint(cfg["paths"]["model_dir"]).dims
    pc = cfg["probe"]
    dom, gen = sets["domain"], sets["general"]
    # every sequence goes through the model (the saved layer inputs are the training cache); the general set is
    # already limited when it is built
    dom_idx, gen_idx = list(range(len(dom))), list(range(len(gen)))
    if dom.offsets[-1] > pc["max_domain_tokens"]:
        print(f"note: domain set has {dom.offsets[-1]} tokens (> max_domain_tokens); all are processed")
    log_f = JsonlLog(os.path.join(cfg["paths"]["logs"], "probe.jsonl"))

    def log(msg):
        print(msg, flush=True)
        log_f(msg=msg)

    log(f"probe: domain {len(dom_idx)} seqs / {sum(dom.lengths[i] for i in dom_idx)} tok, general {len(gen_idx)} seqs"
        f" / {sum(gen.lengths[i] for i in gen_idx)} tok")
    out_dir = os.path.join(cfg["paths"]["work_dir"], "probe")
    cands = [c for c in pc["cache_candidates"] if c <= d.n_layers]
    # stats on domain train+val (eval held out) and on the general 'probe' half; hidden for everything
    st = run_bottom(cfg, {"domain": (dom, dom_idx), "general": (gen, gen_idx)}, d.n_layers, out_dir, save_at=cands,
                    stats_sets={"domain": ["train", "val"], "general": ["probe"]}, mb_tokens=pc["mb_tokens"], snapshot_every=pc["state_every"],
                    log=log, resume=not args.fresh)
    stats = {n: s.state() for n, s in st.items()}
    torch.save({**stats, "dims": d.__dict__}, os.path.join(cfg["paths"]["artifacts"], "routing_stats.pt"))
    kl, jsd = write_report(cfg, stats, d, os.path.join(cfg["paths"]["reports"], "probe.md"),
                           os.path.join(cfg["paths"]["reports"], "probe_divergence.png"))
    top = kl.argsort(descending=True)[:5].tolist()
    append_progress(cfg, f"### probe done\n- layers with the largest KL(dom||gen): {top}\n"
                         f"- caches saved for L in {cands} under {out_dir}\n- report: reports/probe.md")
    log("probe done")


if __name__ == "__main__":
    main()
