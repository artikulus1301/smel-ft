"""Mixed-precision experts GGUF (smel/mixed_experts.py): hot/cold sets, id maps, coverage, and the GGUF layout written
by build_mixed_gguf on a small synthetic MoE GGUF (cold experts copied, so every byte can be checked)."""
import numpy as np
import pytest

from smel.mixed_experts import build_mixed_gguf, choose_sets, coverage, id_maps, parse_hot_layer

gguf = pytest.importorskip("gguf")
QT = gguf.GGMLQuantizationType
N_EXP, N_USED, H, I = 8, 2, 32, 16


def test_choose_sets_by_counts_and_overrides():
    counts = {0: np.array([5, 1, 9, 0, 7, 3, 3, 8.]), 1: np.arange(8.)}
    hot, cold = choose_sets(N_EXP, [0, 1], 3, parse_hot_layer(["1:5"]), counts)
    assert hot[0].tolist() == [2, 4, 7] and cold[0].tolist() == [0, 1, 3, 5, 6]   # counts 9, 8, 7
    assert hot[1].tolist() == [3, 4, 5, 6, 7] and cold[1].tolist() == [0, 1, 2]
    a, _ = choose_sets(N_EXP, [0], 3, random_seed=1)
    b, _ = choose_sets(N_EXP, [0], 3, random_seed=1)
    assert a[0].tolist() == b[0].tolist()
    with pytest.raises(ValueError):
        choose_sets(N_EXP, [0], N_EXP, counts=counts)                # nothing left for the cold set
    with pytest.raises(ValueError):
        choose_sets(N_EXP, [0], 3)                                   # neither counts nor a seed


def test_id_maps():
    m = id_maps(N_EXP, np.array([1, 4, 6]), np.array([0, 2, 3, 5, 7]))
    assert m["map_hot"][:, 0].tolist() == [3, 0, 3, 3, 1, 3, 2, 3]  # cold experts -> n_hot (3)
    assert m["map_cold"][:, 0].tolist() == [0, 5, 1, 2, 5, 3, 5, 4]  # hot experts -> n_cold (5)
    assert m["mask_hot"][:, 0].tolist() == [0, 1, 0, 0, 1, 0, 1, 0]


def test_coverage():
    rank = {0: np.array([10., 1, 5, 0])}
    per_layer, total = coverage(rank, {0: np.array([4., 1, 3, 2])}, 2)   # top-2 of rank: experts 0 and 2
    assert per_layer[0] == pytest.approx(0.7) and total == pytest.approx(0.7)


def _source_gguf(path):
    rng = np.random.default_rng(0)
    w = gguf.GGUFWriter(str(path), arch="qwen35moe")
    w.add_uint32("qwen35moe.expert_count", N_EXP)
    w.add_uint32("qwen35moe.expert_used_count", N_USED)
    w.add_tensor("token_embd.weight", rng.normal(size=(10, H)).astype(np.float32))
    for l in (0, 1):
        gate = rng.normal(size=(N_EXP, I, H)).astype(np.float32)      # quantized experts: hot/cold slice bytes
        w.add_tensor(f"blk.{l}.ffn_gate_exps.weight", gguf.quants.quantize(gate, QT.Q8_0), raw_dtype=QT.Q8_0)
        w.add_tensor(f"blk.{l}.ffn_up_exps.weight", rng.normal(size=(N_EXP, I, H)).astype(np.float32))
        w.add_tensor(f"blk.{l}.ffn_down_exps.weight", rng.normal(size=(N_EXP, H, I)).astype(np.float32))
        w.add_tensor(f"blk.{l}.attn_norm.weight", rng.normal(size=(H,)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _bytes(r, name):
    t = next(t for t in r.tensors if t.name == name)
    return t, np.asarray(r.data[t.data_offset:t.data_offset + t.n_bytes])


@pytest.mark.parametrize("n_zero", [0, N_USED])
def test_build_mixed_gguf_layout(tmp_path, n_zero):
    src, out = tmp_path / "src.gguf", tmp_path / "mixed.gguf"
    _source_gguf(src)
    hot = {0: np.array([1, 4, 6]), 1: np.array([0, 2, 3, 5, 7])}
    cold = {l: np.setdiff1d(np.arange(N_EXP), h) for l, h in hot.items()}
    build_mixed_gguf(str(src), str(out), hot, cold, {"gate": "copy", "up": "copy", "down": "copy"}, n_zero=n_zero,
                     log=lambda s: None)
    rs, ro = gguf.GGUFReader(str(src)), gguf.GGUFReader(str(out))
    assert json_field(ro, "smel.mixed_experts.hot_per_layer") == {"0": 3, "1": 5}
    for name in ("token_embd.weight", "blk.0.attn_norm.weight", "blk.1.attn_norm.weight"):
        assert _bytes(rs, name)[1].tobytes() == _bytes(ro, name)[1].tobytes()
    for l in (0, 1):
        for p in ("gate", "up", "down"):
            ts, bs = _bytes(rs, f"blk.{l}.ffn_{p}_exps.weight")
            per_expert = bs.reshape(N_EXP, -1)
            for kind, sel in (("", hot[l]), ("_cold", cold[l])):
                t, b = _bytes(ro, f"blk.{l}.ffn_{p}_exps{kind}.weight")
                assert t.tensor_type == ts.tensor_type and int(t.shape[-1]) == len(sel) + n_zero
                got = b.reshape(len(sel) + n_zero, -1)
                assert got[:len(sel)].tobytes() == per_expert[sel].tobytes()   # the experts' own bytes
                assert not got[len(sel):].any()                                # optional zero experts
        maps = id_maps(N_EXP, hot[l], cold[l])
        for k in ("map_hot", "map_cold", "mask_hot"):
            t, _ = _bytes(ro, f"blk.{l}.ffn_exp_{k}.weight")
            assert np.asarray(t.data).reshape(-1).tolist() == maps[k][:, 0].tolist()


def json_field(r, key):
    import json
    return json.loads(r.get_field(key).contents())
