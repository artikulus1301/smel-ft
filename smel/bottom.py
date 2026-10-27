"""Layer-by-layer pass of token sets through layers [0, L) of the original bf16 model.

Each layer is read once; every microbatch of every set goes through it; the outputs go to a second memmap (ping-pong),
so an interrupted run resumes from the last finished layer. Optionally collects router statistics and saves the layer
inputs at given cut points (these are exactly the caches the training needs)."""
import hashlib
import json
import os
import time

import numpy as np
import torch

from .layer_runner import layer_forward, make_ctx
from .model_io import Checkpoint
from .pipeline import HiddenStore, mb_rows, microbatches, response_mask
from .stream import CheckpointSource


def sets_signature(sets):
    """Identity of the processed token sets (which sequences, which token ids)."""
    out = {}
    for name, (ts, idx) in sets.items():
        rows = np.concatenate([np.arange(ts.offsets[i], ts.offsets[i + 1]) for i in idx])
        out[name] = {"n_seq": len(idx), "N": int(ts.offsets[-1]),
                     "sha1": hashlib.sha1(ts.ids[rows].tobytes()).hexdigest()}
    return out


class RouterStats:
    def __init__(self, n_layers, E, views=("all", "resp")):
        self.d = {v: {k: torch.zeros(n_layers, E, dtype=torch.float64) for k in ("prob_sum", "gate_mass", "topk_cnt")}
                  for v in views}
        self.ntok = {v: torch.zeros(n_layers, dtype=torch.float64) for v in views}

    def add(self, view, l, probs, topv, topi, mask):
        if mask is not None:
            probs, topv, topi = probs[mask], topv[mask], topi[mask]
        if probs.shape[0] == 0:
            return
        E = probs.shape[1]
        d = self.d[view]
        d["prob_sum"][l] += probs.double().sum(0).cpu()
        d["gate_mass"][l] += torch.zeros(E, device=probs.device, dtype=torch.float64).scatter_add_(
            0, topi.reshape(-1), topv.reshape(-1).double()).cpu()
        d["topk_cnt"][l] += torch.bincount(topi.reshape(-1), minlength=E).double().cpu()
        self.ntok[view][l] += probs.shape[0]

    def state(self):
        return {"d": self.d, "ntok": self.ntok}

    def load(self, st):
        self.d, self.ntok = st["d"], st["ntok"]


def run_bottom(cfg, sets: dict, stop_layer: int, out_dir: str, save_at=(), stats_sets=None, mb_tokens=4096,
               log=print, resume=True, snapshot_every=5):
    """sets: {name: (TokenSet, seq_indices)}. stats_sets: {name: split labels whose tokens count in the stats}.
    Saves out_dir/L{l}/h_{name}.npy for l in save_at (input of layer l). Returns {name: RouterStats} or None.
    Hidden states stay in RAM; every `snapshot_every` layers a resumable snapshot is written to out_dir."""
    ck = Checkpoint(cfg["paths"]["model_dir"])
    d = ck.dims
    os.makedirs(out_dir, exist_ok=True)
    state_path = os.path.join(out_dir, "state.json")
    stats = {n: RouterStats(d.n_layers, d.n_experts) for n in stats_sets} if stats_sets else None
    H, mbs = {}, {}  # hidden states of every set live in RAM (bf16, ~4 KB/token)
    start = 0
    sig = sets_signature(sets)
    st = json.load(open(state_path)) if (resume and os.path.exists(state_path)) else None
    if st is not None and (st.get("sig") != sig or st.get("stop_layer") != stop_layer):
        log("token sets / target changed since the saved state -> starting fresh")
        st = None
    for name, (ts, idx) in sets.items():
        mbs[name] = microbatches(ts, idx, mb_tokens)
    snap = lambda name: os.path.join(out_dir, f"snapshot_{name}.npy")  # noqa: E731
    if st is not None:
        start = st["next_layer"]
        for name in sets:
            H[name] = HiddenStore(snap(name)).read(slice(None))
        if stats is not None and os.path.exists(os.path.join(out_dir, "stats.pt")):
            saved = torch.load(os.path.join(out_dir, "stats.pt"))
            for n, s in stats.items():
                s.load(saved[n])
        log(f"resume from layer {start}")
    else:
        for name, (ts, idx) in sets.items():
            H[name] = torch.zeros(int(ts.offsets[-1]), d.hidden, dtype=torch.bfloat16)
            rows = torch.from_numpy(mb_rows(ts, idx))
            H[name][rows] = ck.embed(torch.from_numpy(ts.ids[rows.numpy()].astype(np.int64)))

    def save_h(name, path):
        hs = HiddenStore(path + ".tmp.npy", H[name].shape[0], d.hidden, mode="w+")
        hs.mm[:] = H[name].view(torch.uint16).numpy()
        hs.flush()
        del hs
        os.replace(path + ".tmp.npy", path)

    src = CheckpointSource(ck, "cuda")
    t_all = time.perf_counter()
    for l in range(start, stop_layer):
        if l in save_at:
            dst = os.path.join(out_dir, f"L{l}")
            os.makedirs(dst, exist_ok=True)
            for name in sets:
                save_h(name, os.path.join(dst, f"h_{name}.npy"))
        t0 = time.perf_counter()
        LW = src.get(l)
        if l + 1 < stop_layer:
            src.prefetch(l + 1)
        t_load = time.perf_counter() - t0
        ntok = 0
        for name, (ts, idx) in sets.items():
            resp_all = response_mask(ts.tgt)
            in_stats = None
            if stats is not None and name in stats_sets:
                labels = np.array(ts.split)
                seq_in = np.isin(labels, stats_sets[name])
                lens = np.diff(ts.offsets)
                in_stats = np.repeat(seq_in, lens)
            for mb in mbs[name]:
                rows = mb_rows(ts, mb)
                rows_t = torch.from_numpy(rows)
                h = H[name][rows_t].cuda(non_blocking=True)
                ctx = make_ctx([ts.lengths[i] for i in mb], d, "cuda", torch.bfloat16)
                cb = None
                if in_stats is not None:
                    m_all = torch.from_numpy(in_stats[rows]).cuda()
                    m_resp = m_all & torch.from_numpy(resp_all[rows]).cuda()

                    def cb(p, v, t, _l=l, ma=m_all, mr=m_resp, S=stats[name]):
                        S.add("all", _l, p, v, t, ma)
                        S.add("resp", _l, p, v, t, mr)
                with torch.no_grad():
                    out = layer_forward(h, LW, d, ctx, None, cb)
                H[name][rows_t] = out.cpu()
                ntok += len(rows)
        dt = time.perf_counter() - t0
        if (l + 1) % snapshot_every == 0 and l + 1 < stop_layer:
            for name in sets:
                save_h(name, snap(name))
            if stats is not None:
                torch.save({n: s.state() for n, s in stats.items()}, os.path.join(out_dir, "stats.pt"))
            json.dump({"stop_layer": stop_layer, "next_layer": l + 1, "sig": sig}, open(state_path, "w"))
        log(f"layer {l:2d} ({d.layer_types[l][:4]}): {dt:6.1f}s (wait/load {t_load:4.1f}s), {ntok/dt:8.0f} tok/s, "
            f"peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GB")
    if stop_layer in save_at:
        dst = os.path.join(out_dir, f"L{stop_layer}")
        os.makedirs(dst, exist_ok=True)
        for name in sets:
            save_h(name, os.path.join(dst, f"h_{name}.npy"))
    if stats is not None:
        torch.save({n: s.state() for n, s in stats.items()}, os.path.join(out_dir, "stats.pt"))
    json.dump(sig, open(os.path.join(out_dir, "sets_sig.json"), "w"))
    for name in sets:  # the run is complete; drop the resume snapshot (our own temporary file)
        if os.path.exists(snap(name)):
            os.remove(snap(name))
    if os.path.exists(state_path):
        os.remove(state_path)
    src.close()
    log(f"bottom pass done in {time.perf_counter() - t_all:.0f}s, load {src.stats['load_s']:.0f}s, "
        f"{src.stats['bytes']/2**30:.1f} GB read")
    return stats
