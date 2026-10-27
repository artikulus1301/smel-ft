"""Tokenization with the model's chat template, loss masks on assistant answers, splits and length-bucketed packing."""
import hashlib
import json
import random

import numpy as np

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatSpec:
    """Markers of the assistant turn in a rendered chat template."""
    header: str        # text that opens an assistant turn
    end: str           # text that closes it (supervised, so the model learns to stop)
    think_open: str    # reasoning block at the start of an answer (excluded from the loss unless think_in_loss)
    think_close: str
    empty_think: str   # what the template puts before an answer when thinking is off; "" = nothing to add


QWEN_SPEC = ChatSpec("<|im_start|>assistant\n", "<|im_end|>", "<think>", "</think>", "")
# Gemma 4: history turns are rendered without a thought channel, but generation with thinking off starts the
# answer after an empty one ("<|channel>thought\n<channel|>") - training adds it so both look the same
GEMMA4_SPEC = ChatSpec("<|turn>model\n", "<turn|>", "<|channel>", "<channel|>", "<|channel>thought\n<channel|>")


def chat_spec(tok) -> ChatSpec:
    t = getattr(tok, "chat_template", None) or ""
    if "<|turn>" in t:
        return GEMMA4_SPEC
    if "<|im_start|>" in t:
        return QWEN_SPEC
    raise ValueError("unknown chat template: add a ChatSpec for it in smel/data.py")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def add_empty_think(text: str, spec: ChatSpec) -> str:
    if not spec.empty_think:
        return text
    out, pos = [], 0
    while (i := text.find(spec.header, pos)) >= 0:
        s = i + len(spec.header)
        out.append(text[pos:s])
        if not text.startswith(spec.think_open, s):
            out.append(spec.empty_think)
        pos = s
    out.append(text[pos:])
    return "".join(out)


def assistant_spans(text: str, think_in_loss: bool = False, spec: ChatSpec = QWEN_SPEC):
    """Character spans of supervised text: each assistant answer including its end marker. A reasoning block at the
    start of the answer (e.g. Qwen's '<think>...</think>\\n\\n') is excluded unless think_in_loss."""
    spans, pos = [], 0
    while True:
        i = text.find(spec.header, pos)
        if i < 0:
            break
        s = i + len(spec.header)
        e = text.find(spec.end, s)
        e = len(text) if e < 0 else e + len(spec.end)
        if not think_in_loss and text.startswith(spec.think_open, s):
            te = text.find(spec.think_close, s)
            if 0 <= te < e:
                s = te + len(spec.think_close)
                while s < e and text[s] == "\n":
                    s += 1
        spans.append((s, e))
        pos = e
    return spans


def encode_chat(tok, messages, think_in_loss=False, max_len=None):
    """Returns (ids int32 [T], targets int32 [T]) where targets[t] = ids[t+1] if token t+1 is supervised else -100."""
    spec = chat_spec(tok)
    text = add_empty_think(tok.apply_chat_template(messages, tokenize=False), spec)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = np.asarray(enc["input_ids"], dtype=np.int32)
    offs = enc["offset_mapping"]
    spans = assistant_spans(text, think_in_loss, spec)
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
