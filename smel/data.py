"""Tokenization with the model's chat template, loss masks on assistant answers, splits and length-bucketed packing."""
import hashlib
import json
import random

import numpy as np

ASSIST_HDR = "<|im_start|>assistant\n"
END = "<|im_end|>"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def assistant_spans(text: str, think_in_loss: bool = False):
    """Character spans of supervised text: each assistant answer including its <|im_end|>. The template's
    '<think>...</think>\\n\\n' prefix is excluded unless think_in_loss."""
    spans, pos = [], 0
    while True:
        i = text.find(ASSIST_HDR, pos)
        if i < 0:
            break
        s = i + len(ASSIST_HDR)
        e = text.find(END, s)
        e = len(text) if e < 0 else e + len(END)
        if not think_in_loss and text.startswith("<think>", s):
            te = text.find("</think>", s)
            if 0 <= te < e:
                s = te + len("</think>")
                while s < e and text[s] == "\n":
                    s += 1
        spans.append((s, e))
        pos = e
    return spans


def encode_chat(tok, messages, think_in_loss=False, max_len=None):
    """Returns (ids int32 [T], targets int32 [T]) where targets[t] = ids[t+1] if token t+1 is supervised else -100."""
    text = tok.apply_chat_template(messages, tokenize=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = np.asarray(enc["input_ids"], dtype=np.int32)
    offs = enc["offset_mapping"]
    spans = assistant_spans(text, think_in_loss)
    sup = np.zeros(len(ids), dtype=bool)
    j = 0
    for t, (a, b) in enumerate(offs):
        while j < len(spans) and a >= spans[j][1]:
            j += 1
        if j < len(spans) and spans[j][0] <= a < spans[j][1] and b > a:
            sup[t] = True
    if max_len is not None and len(ids) > max_len:
        ids, sup = ids[:max_len], sup[:max_len]
    tgt = np.full(len(ids), -100, dtype=np.int32)
    tgt[:-1] = np.where(sup[1:], ids[1:], -100)
    return ids, tgt


def split_rows(rows, n_val: int, n_eval: int, seed: int = 0):
    """Deterministic split by content hash order (stable when rows are appended)."""
    keyed = sorted(rows, key=lambda r: hashlib.sha1((json.dumps(r, ensure_ascii=False) + str(seed))
                                                    .encode()).hexdigest())
    return {"eval": keyed[:n_eval], "val": keyed[n_eval:n_eval + n_val], "train": keyed[n_eval + n_val:]}


def pack(lengths, max_tokens: int, shuffle_seed=None):
    """First-fit-decreasing packing of sequence indices into microbatches of at most max_tokens tokens."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    bins, room = [], []
    for i in order:
        n = lengths[i]
        for b in range(len(bins)):
            if room[b] >= n:
                bins[b].append(i)
                room[b] -= n
                break
        else:
            bins.append([i])
            room.append(max_tokens - n)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(bins)
    return bins


def make_steps(microbatches, lengths, tokens_per_step: int):
    """Group microbatches into optimizer steps of ~tokens_per_step tokens."""
    steps, cur, n = [], [], 0
    for mb in microbatches:
        t = sum(lengths[i] for i in mb)
        if cur and n + t > tokens_per_step:
            steps.append(cur)
            cur, n = [], 0
        cur.append(mb)
        n += t
    if cur:
        steps.append(cur)
    return steps
