"""Bottom-layer pass: saved layer inputs == direct forward; an interrupted run resumes to the same result."""
import numpy as np
import pytest
import torch

from helpers import layers_from_hf
from smel.bottom import run_bottom
from smel.layer_runner import LayerMajorRunner, make_ctx
from smel.pipeline import HiddenStore, TokenSet
from smel.stream import DictSource


def _tokens():
    g = np.random.default_rng(0)
    lens = [40, 90, 17, 130, 61]
    ids = g.integers(0, 512, sum(lens)).astype(np.int32)
    tgt = np.full(len(ids), -100, dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    for i in range(len(lens)):
        a, b = offsets[i], offsets[i + 1]
        tgt[a + 5:b - 1] = ids[a + 6:b]
    return TokenSet(ids, tgt, offsets, ["train", "val", "train", "eval", "train"])


class Crash(Exception):
    pass


def test_run_bottom_matches_and_resumes(tiny_ckpt, model16, tmp_path):
    cfg = {"paths": {"model_dir": tiny_ckpt}}
    ts = _tokens()
    sets = {"domain": (ts, list(range(len(ts))))}
    ref_dir, crash_dir = str(tmp_path / "ref"), str(tmp_path / "crash")
    st = run_bottom(cfg, sets, 6, ref_dir, save_at=[3, 6], stats_sets={"domain": ["train"]}, mb_tokens=128)
    h6 = HiddenStore(f"{ref_dir}/L6/h_domain.npy").read(slice(None))

    # direct forward of layers 0..5 per sequence
    from helpers import dims_of
    d = dims_of(model16)
    run = LayerMajorRunner(d, DictSource(layers_from_hf(model16, layers=range(6))), list(range(6)))
    emb = model16.model.embed_tokens.weight.detach()
    hs = [emb[torch.from_numpy(ts.seq(i)[0].astype(np.int64)).cuda()] for i in range(len(ts))]
    ctxs = [make_ctx([ts.lengths[i]], d, "cuda", torch.bfloat16) for i in range(len(ts))]
    out, _ = run.forward(hs, ctxs)
    direct = torch.cat(out).cpu()
    # different microbatch packing -> bf16 noise only
    assert ((h6.float() - direct.float()).norm() / direct.float().norm()) < 3e-2
    assert st["domain"].ntok["all"][0] == sum(ts.lengths[i] for i in (0, 2, 4))

    def crashing_log(msg, n=[0]):
        if msg.startswith("layer"):
            n[0] += 1
            if n[0] == 5:
                raise Crash()

    with pytest.raises(Crash):
        run_bottom(cfg, sets, 6, crash_dir, save_at=[3, 6], stats_sets={"domain": ["train"]}, mb_tokens=128,
                   log=crashing_log, snapshot_every=2)
    msgs = []
    st2 = run_bottom(cfg, sets, 6, crash_dir, save_at=[3, 6], stats_sets={"domain": ["train"]}, mb_tokens=128,
                     log=msgs.append, snapshot_every=2)
    assert any("resume from layer 4" in m for m in msgs), msgs
    h6b = HiddenStore(f"{crash_dir}/L6/h_domain.npy").read(slice(None))
    assert torch.equal(h6b, h6)
    for k in ("prob_sum", "gate_mass", "topk_cnt"):
        assert torch.allclose(st2["domain"].d["resp"][k], st["domain"].d["resp"][k])
