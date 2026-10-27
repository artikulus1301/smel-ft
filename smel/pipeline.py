"""Shared pipeline utilities: config, token sets, hidden-state stores, bottom-layer runs, monitoring helpers."""
import json
import os
import subprocess
import time

import numpy as np
import torch
import yaml

from .data import encode_chat, load_jsonl, pack, split_rows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None, overrides=None):
    path = path or os.path.join(ROOT, "configs", "default.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in (overrides or {}).items():
        d = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d.setdefault(kk, {})
        if isinstance(v, str):
            v = yaml.safe_load(v)
            if isinstance(v, str):  # YAML 1.1 reads "3e-3" as a string
                try:
                    v = float(v)
                except ValueError:
                    pass
        d[ks[-1]] = v
    paths = cfg["paths"]
    if not paths.get("tokenizer"):
        paths["tokenizer"] = paths["model_dir"]
    paths.setdefault("progress", "logs/progress.md")
    paths.setdefault("selection", "configs/selection.yaml")   # written by select_experts.py, read by the rest
    for k, p in paths.items():
        if isinstance(p, str) and not os.path.isabs(p):
            paths[k] = os.path.normpath(os.path.join(ROOT, p))
    for k in ("work_dir", "artifacts", "reports", "logs"):
        os.makedirs(paths[k], exist_ok=True)
    cfg.setdefault("llamacpp", {})
    cfg["llamacpp"]["dir"] = cfg["llamacpp"].get("dir") or os.environ.get("SMEL_LLAMA_CPP_DIR")
    return cfg


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = v
    return out


# ----------------------------------------------------------------------------------------------- token sets
class TokenSet:
    """Concatenated tokenized sequences: ids/tgt int32 [N], offsets int64 [n+1], split label per sequence."""

    def __init__(self, ids, tgt, offsets, split):
        self.ids, self.tgt, self.offsets, self.split = ids, tgt, offsets, list(split)

    @property
    def lengths(self):
        return np.diff(self.offsets).tolist()

    def __len__(self):
        return len(self.split)

    def seq(self, i):
        a, b = self.offsets[i], self.offsets[i + 1]
        return self.ids[a:b], self.tgt[a:b]

    def indices(self, split):
        return [i for i, s in enumerate(self.split) if s == split]

    def subset(self, idx):
        idx = list(idx)
        lens = [self.lengths[i] for i in idx]
        rows = np.concatenate([np.arange(self.offsets[i], self.offsets[i + 1]) for i in idx])
        offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        return TokenSet(self.ids[rows], self.tgt[rows], offsets, [self.split[i] for i in idx])

    def save(self, path):
        np.savez(path, ids=self.ids, tgt=self.tgt, offsets=self.offsets, split=np.array(self.split))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["ids"], z["tgt"], z["offsets"], z["split"].tolist())


def _rows_to_messages(r):
    if "messages" in r:
        return r["messages"], True
    return r["text"], False


def tokenize_rows(tok, rows, split_labels, think_in_loss, max_len):
    ids_l, tgt_l, lens = [], [], []
    for r in rows:
        msgs, is_chat = _rows_to_messages(r)
        if is_chat:
            ids, tgt = encode_chat(tok, msgs, think_in_loss, max_len)
        else:
            ids = np.asarray(tok(msgs, add_special_tokens=False)["input_ids"][:max_len], dtype=np.int32)
            tgt = np.full(len(ids), -100, dtype=np.int32)
            tgt[:-1] = ids[1:]
        ids_l.append(ids)
        tgt_l.append(tgt)
        lens.append(len(ids))
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    return TokenSet(np.concatenate(ids_l).astype(np.int32), np.concatenate(tgt_l).astype(np.int32), offsets,
                    split_labels)


def _limit(ts, idx, max_tokens):
    out, n = [], 0
    for i in idx:
        if n + ts.lengths[i] > max_tokens:
            break
        out.append(i)
        n += ts.lengths[i]
    return out


def build_token_sets(cfg, tok):
    """Domain: train/val/eval by deterministic split (all rows). General: probe/eval halves, limited to
    probe.max_general_tokens (+ half of that for eval). Cached under work_dir."""
    out = {}
    dc = cfg["data"]
    paths = {"domain": cfg["paths"]["domain_data"], "general": cfg["paths"]["general_data"]}
    key = json.dumps([dc, cfg["probe"]["max_general_tokens"], cfg["paths"]["tokenizer"], paths], sort_keys=True)
    for name, p in paths.items():
        cache = os.path.join(cfg["paths"]["work_dir"], f"tokens_{name}.npz")
        side = cache + ".json"
        if (os.path.exists(cache) and os.path.exists(side) and os.path.getmtime(cache) > os.path.getmtime(p)
                and json.load(open(side)).get("key") == key):
            out[name] = TokenSet.load(cache)
            continue
        rows = load_jsonl(p)
        if name == "domain":
            sp = split_rows(rows, dc["n_val"], dc["n_eval"], dc["seed"])
            rows = sp["train"] + sp["val"] + sp["eval"]
            labels = ["train"] * len(sp["train"]) + ["val"] * len(sp["val"]) + ["eval"] * len(sp["eval"])
        else:
            n_eval = int(len(rows) * dc["general_eval_frac"])
            sp = split_rows(rows, 0, n_eval, dc["seed"])
            rows = sp["train"] + sp["eval"]
            labels = ["probe"] * len(sp["train"]) + ["eval"] * len(sp["eval"])
        ts = tokenize_rows(tok, rows, labels, dc["think_in_loss"], dc["max_len"])
        if name == "general":
            mg = cfg["probe"]["max_general_tokens"]
            ts = ts.subset(_limit(ts, ts.indices("probe"), mg) + _limit(ts, ts.indices("eval"), mg // 2))
        ts.save(cache)
        json.dump({"key": key, "n_seq": len(ts), "tokens": int(ts.offsets[-1])}, open(side, "w"))
        out[name] = ts
    return out


def response_mask(tgt: np.ndarray) -> np.ndarray:
    """True at positions whose token is supervised (part of an answer): tgt[t-1] != -100."""
    m = np.zeros(len(tgt), dtype=bool)
    m[1:] = tgt[:-1] != -100
    return m


# ----------------------------------------------------------------------------------------------- hidden stores
class HiddenStore:
    """bf16 hidden states [N, H] in a .npy memmap (stored as uint16)."""

    def __init__(self, path, N=None, H=None, mode="r"):
        self.path = path
        if mode == "w+":
            self.mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=(N, H))
        else:
            self.mm = np.load(path, mmap_mode=mode)

    def read(self, rows) -> torch.Tensor:
        a = self.mm[rows]
        if not a.flags.owndata:  # a slice is a view into the (read-only) map: copy it
            a = np.array(a)
        return torch.from_numpy(a).view(torch.bfloat16)

    def write(self, rows: np.ndarray, h: torch.Tensor):
        self.mm[rows] = h.detach().to("cpu", torch.bfloat16).view(torch.uint16).numpy()

    def flush(self):
        self.mm.flush()


def mb_rows(ts: TokenSet, seq_idx):
    return np.concatenate([np.arange(ts.offsets[i], ts.offsets[i + 1]) for i in seq_idx])


def microbatches(ts: TokenSet, seq_idx, mb_tokens, seed=None):
    lens = [ts.lengths[i] for i in seq_idx]
    bins = pack(lens, mb_tokens, shuffle_seed=seed)
    return [[seq_idx[j] for j in b] for b in bins]


# ----------------------------------------------------------------------------------------------- misc
def gpu_stats():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,power.draw,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        t, m, p, u = [x.strip() for x in out.strip().split(",")]
        return {"gpu_temp": float(t), "gpu_mem_used_mb": float(m), "gpu_power_w": float(p), "gpu_util": float(u)}
    except Exception:
        return {}


def rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**30
    except Exception:
        return float("nan")


class JsonlLog:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, **kw):
        kw.setdefault("time", time.time())
        self.f.write(json.dumps(kw, ensure_ascii=False) + "\n")
        self.f.flush()


def append_progress(cfg, text):
    p = cfg["paths"].get("progress") or os.path.join(ROOT, "PROGRESS.md")
    with open(p, "a", encoding="utf-8") as f:
        f.write("\n" + text.rstrip() + "\n")
