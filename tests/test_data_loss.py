"""Loss masks (only assistant answers), packing, chunked CE (bf16 / int8 lm_head)."""
import os

import pytest
import torch
import torch.nn.functional as F

from smel.data import encode_chat, load_jsonl, pack
from smel.loss import LMHead, chunked_ce
from smel.quant import rel_err

# optional: your own chat jsonl to check the masks on real data
DATA = os.environ.get("SMEL_TEST_DATA", "")


def _supervised_text(tok, ids, tgt):
    import numpy as np
    sup = np.zeros(len(ids), dtype=bool)
    sup[1:] = tgt[:-1] != -100
    pieces, cur = [], []
    for i, s in enumerate(sup):
        if s:
            cur.append(int(ids[i]))
        elif cur:
            pieces.append(tok.decode(cur))
            cur = []
    if cur:
        pieces.append(tok.decode(cur))
    return pieces


def test_masks_only_assistant_answers(tokenizer):
    msgs = [{"role": "system", "content": "Ты Азуса."}, {"role": "user", "content": "Привет"},
            {"role": "assistant", "content": "Ну привет."}, {"role": "user", "content": "Как дела?"},
            {"role": "assistant", "content": "Нормально, отстань."}]
    ids, tgt = encode_chat(tokenizer, msgs)
    pieces = _supervised_text(tokenizer, ids, tgt)
    assert pieces == ["Ну привет.<|im_end|>", "Нормально, отстань.<|im_end|>"], pieces
    # targets are next tokens
    for t in range(len(ids) - 1):
        assert tgt[t] == -100 or tgt[t] == ids[t + 1]


def test_think_block_optional(tokenizer):
    msgs = [{"role": "user", "content": "Привет"}, {"role": "assistant", "content": "Ну привет."}]
    ids, tgt = encode_chat(tokenizer, msgs, think_in_loss=True)
    assert _supervised_text(tokenizer, ids, tgt)[0].startswith("<think>")


@pytest.mark.skipif(not DATA or not os.path.exists(DATA), reason="set SMEL_TEST_DATA to a chat jsonl")
def test_masks_on_real_dataset(tokenizer):
    rows = load_jsonl(DATA)[:200]
    for r in rows:
        ids, tgt = encode_chat(tokenizer, r["messages"])
        got = _supervised_text(tokenizer, ids, tgt)
        want = [m["content"].strip() + "<|im_end|>" for m in r["messages"] if m["role"] == "assistant"]
        assert got == want


def test_pack_respects_budget():
    lens = [5, 300, 120, 700, 33, 1000, 999, 2, 64]
    bins = pack(lens, 1024)
    assert sorted(i for b in bins for i in b) == list(range(len(lens)))
    assert all(sum(lens[i] for i in b) <= 1024 for b in bins)


@pytest.mark.parametrize("int8", [False, True])
def test_chunked_ce(int8):
    torch.manual_seed(0)
    V, H, N = 5000, 128, 300
    W = (torch.randn(V, H, device="cuda") * 0.05).bfloat16()
    head = LMHead.from_weight(W, int8=int8, vocab_chunk=1024)
    Wref = head.rows(0, V, torch.float32)
    h = torch.randn(N, H, device="cuda").bfloat16().requires_grad_(True)
    tgt = torch.randint(0, V, (N,), device="cuda")
    loss = chunked_ce(h, head, tgt, denom=N, token_chunk=64)
    loss.backward()
    hr = h.detach().float().requires_grad_(True)
    ref = F.cross_entropy(hr @ Wref.t(), tgt)
    ref.backward()
    assert abs(float(loss) - float(ref)) < 1e-2
    assert rel_err(h.grad, hr.grad) < 2e-2
    if int8:
        assert rel_err(Wref, W.float()) < 1e-2
