"""Checkpoint loading (real VL layout, mmap) and streaming sources with prefetch == resident weights."""
import pytest
import torch
from safetensors.torch import save_file

from helpers import dims_of, layers_from_hf
from smel.layer_runner import LayerMajorRunner, make_ctx
from smel.model_io import Checkpoint
from smel.pack import pack_layer
from smel.stream import CheckpointSource, DictSource, PackedSource


def _run(d, src, layers, seqs, dtype=torch.bfloat16, embed=None):
    run = LayerMajorRunner(d, src, layers)
    hs = [embed[ids].to(dtype) for ids, _ in seqs]
    ctxs = [make_ctx([len(ids)], d, "cuda", dtype) for ids, _ in seqs]
    hs, _ = run.forward(hs, ctxs)
    return torch.cat(hs)


def test_checkpoint_layout_and_dims(tiny_ckpt, model16):
    ck = Checkpoint(tiny_ckpt)
    assert ck.prefix == "model.language_model."
    d = ck.dims
    assert d == dims_of(model16)
    names = {k.split("layers.0.")[1] for k in ck.layer_names(0)}
    assert "mlp.experts.gate_up_proj" in names and "mlp.gate.weight" in names
    gu = ck.get(ck.prefix + "layers.0.mlp.experts.gate_up_proj")
    rows = ck.get_rows(ck.prefix + "layers.0.mlp.experts.gate_up_proj", [3, 7])
    assert torch.equal(rows, gu[[3, 7]])


def test_checkpoint_source_prefetch_matches_resident(tiny_ckpt, model16, seqs):
    ck = Checkpoint(tiny_ckpt)
    d = ck.dims
    layers = list(range(d.n_layers))
    embed = ck.get(ck.embed_name()).cuda()
    ref = _run(d, DictSource(layers_from_hf(model16)), layers, seqs, embed=embed)
    src = CheckpointSource(ck, "cuda")
    ours = _run(d, src, layers, seqs, embed=embed)
    assert src.stats["loads"] == d.n_layers
    assert torch.equal(ours, ref)
    src.close()


@pytest.mark.parametrize("pin", [False, True])
def test_packed_source_matches_dict(tmp_path, tiny_ckpt, seqs, pin):
    ck = Checkpoint(tiny_ckpt)
    d = ck.dims
    L = 2
    layers = list(range(L, d.n_layers))
    sel = {l: [l, l + 5] for l in layers}
    for l in layers:
        save_file(pack_layer(ck.layer(l), sel[l], 64), str(tmp_path / f"layer_{l:02d}.safetensors"))
    embed = ck.get(ck.embed_name()).cuda()
    src_ck = CheckpointSource(ck, "cuda", selection=sel, frozen="int4")
    ref = _run(d, src_ck, layers, seqs, embed=embed)
    src = PackedSource(str(tmp_path), d, layers, "cuda", pin=pin)
    ours = _run(d, src, layers, seqs, embed=embed)
    assert torch.equal(ours, ref)
    # with the selected experts in bf16 the result differs from all-int4
    allq = CheckpointSource(ck, "cuda", selection={}, frozen="int4")
    assert not torch.equal(_run(d, allq, layers, seqs, embed=embed), ref)
    for s in (src_ck, src, allq):
        s.close()
