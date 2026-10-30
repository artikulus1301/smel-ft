"""Mixed-precision experts for inference: in every MoE layer the most used experts keep the quantization of the source
GGUF ("hot" set) and the others ("cold" set) go into a lower-bit type, re-quantized from the merged bf16 weights with an
importance matrix. The result is a GGUF for a llama.cpp build with inference/llama_cpp_patches/mixed-experts-*.patch.

GGUF layout per MoE layer L (the original blk.L.ffn_{gate,up,down}_exps tensors now hold the hot experts only):
  blk.L.ffn_{gate,up,down}_exps_cold  the cold experts, in the cold type
  blk.L.ffn_exp_map_hot  / _cold      F32 [n_expert]: router expert id -> index in the own set, n_set for the other set
  blk.L.ffn_exp_mask_hot              F32 [n_expert]: 1 for a hot expert, 0 for a cold one
Everything else is copied byte for byte. With cold type "copy" the cold experts keep their bytes too: the file must then
compute exactly what the source computes (the test of the patch).

The routing counts that rank the experts come from llama-imatrix (blk.L.ffn_up_exps.weight.counts)."""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

EXPS = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
COUNTS = re.compile(r"^blk\.(\d+)\.ffn_up_exps\.weight\.counts$")
PROJ = ("gate", "up", "down")


def read_counts(paths):
    """{layer: routing counts [n_expert]} summed over llama-imatrix GGUF files."""
    from gguf import GGUFReader
    out = {}
    for p in paths:
        r = GGUFReader(p)
        for t in r.tensors:
            m = COUNTS.match(t.name)
            if m:
                c = np.asarray(t.data, dtype=np.float64).reshape(-1)
                out[int(m.group(1))] = out.get(int(m.group(1)), 0) + c
        del r
    return dict(sorted(out.items()))


def coverage(rank_counts, eval_counts, n):
    """{layer: share of the eval routings that go to the n most used experts of rank_counts}, and the overall share
    (layers weighted by their routings)."""
    per_layer = {}
    for l, c in eval_counts.items():
        top = np.argsort(-rank_counts[l], kind="stable")[:n]
        per_layer[l] = float(c[top].sum() / max(c.sum(), 1))
    w = np.array([eval_counts[l].sum() for l in per_layer])
    return per_layer, float(np.dot(list(per_layer.values()), w) / max(w.sum(), 1))


def parse_hot_layer(items):
    """['0:192', '1:192'] -> {0: 192, 1: 192}"""
    out = {}
    for s in items or []:
        layer, n = s.split(":")
        out[int(layer)] = int(n)
    return out


def choose_sets(n_expert, layers, hot_default, hot_layer=None, counts=None, random_seed=None):
    """-> (hot, cold): {layer: sorted int array}. Hot = the experts with the highest routing counts of the layer (ties:
    lower id first), or random ones with random_seed."""
    hot_layer = hot_layer or {}
    if counts is None and random_seed is None:
        raise ValueError("counts or random_seed is needed to choose the hot experts")
    rng = np.random.default_rng(random_seed)
    hot, cold = {}, {}
    for l in layers:
        n = hot_layer.get(l, hot_default)
        if not 0 < n < n_expert:
            raise ValueError(f"layer {l}: {n} hot experts of {n_expert}")
        order = rng.permutation(n_expert) if counts is None else np.argsort(-counts[l], kind="stable")
        hot[l], cold[l] = np.sort(order[:n]), np.sort(order[n:])
    return hot, cold


def id_maps(n_expert, hot, cold):
    """The three small tables of a layer ([n_expert, 1], F32 - small integers are exact)."""
    map_hot = np.full((n_expert, 1), len(hot), np.float32)
    map_cold = np.full((n_expert, 1), len(cold), np.float32)
    map_hot[hot, 0] = np.arange(len(hot))
    map_cold[cold, 0] = np.arange(len(cold))
    mask = np.zeros((n_expert, 1), np.float32)
    mask[hot, 0] = 1.0
    return {"map_hot": map_hot, "map_cold": map_cold, "mask_hot": mask}


def build_mixed_gguf(src_path, out_path, hot, cold, cold_types, hf=None, imatrix=None, llama_cpp=None, workers=3,
                     n_zero=0, log=print):
    """Write the mixed GGUF. cold_types: {"gate"|"up"|"down": ggml type name or "copy"}. Re-quantizing cold experts needs
    hf (merged safetensors dir), imatrix (llama-imatrix file used for the source quant) and a llama.cpp build."""
    import gguf
    from gguf import GGMLQuantizationType as QT
    src = gguf.GGUFReader(src_path)
    tensors = list(src.tensors)
    arch = src.get_field("general.architecture").contents()
    n_expert = int(src.get_field(f"{arch}.expert_count").contents())
    used = int(src.get_field(f"{arch}.expert_used_count").contents())
    if n_zero not in (0, used):   # 0, or expert_used_count for the first version of the patch (zero experts)
        raise ValueError(f"n_zero must be 0 or {used}")
    layers = sorted(hot)
    cold_qt = {k: None if cold_types[k].lower() == "copy" else QT[cold_types[k].upper()] for k in PROJ}

    def row_bytes(qt, n_per_row):
        bs, ts = gguf.GGML_QUANT_SIZES[qt]
        assert n_per_row % bs == 0, (qt.name, n_per_row)
        return n_per_row // bs * ts

    def raw(t):
        return np.asarray(src.data[t.data_offset:t.data_offset + t.n_bytes])

    # plan: (name, kind, payload) in file order
    plan = []
    for t in tensors:
        m = EXPS.match(t.name)
        if not m or int(m.group(1)) not in hot:
            plan.append((t.name, "copy", t))
            continue
        l, kind = int(m.group(1)), m.group(2)
        plan.append((t.name, "hot", (t, l)))
        plan.append((f"blk.{l}.ffn_{kind}_exps_cold.weight", "cold", (t, l, kind)))
        if kind == "down":
            plan += [(f"blk.{l}.ffn_exp_map_hot.weight", "map_hot", l), (f"blk.{l}.ffn_exp_map_cold.weight", "map_cold", l),
                     (f"blk.{l}.ffn_exp_mask_hot.weight", "mask_hot", l)]

    w = gguf.GGUFWriter(out_path, arch=arch, endianess=src.endianess)
    for f in src.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]
        w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    w.add_key_value("smel.mixed_experts.cold_type", json.dumps(cold_types), gguf.GGUFValueType.STRING)
    w.add_key_value("smel.mixed_experts.hot_per_layer", json.dumps({str(l): int(len(hot[l])) for l in layers}),
                    gguf.GGUFValueType.STRING)

    for name, kind, p in plan:
        if kind == "copy":
            t = p
            np_shape = list(reversed([int(x) for x in t.shape]))
            if t.tensor_type in (QT.F32, QT.F16, QT.I32):
                dt = {QT.F32: np.float32, QT.F16: np.float16, QT.I32: np.int32}[t.tensor_type]
                w.add_tensor_info(name, np_shape, dt, int(t.n_bytes), raw_dtype=t.tensor_type)
            else:
                np_shape[-1] = row_bytes(t.tensor_type, np_shape[-1])
                w.add_tensor_info(name, np_shape, np.uint8, int(t.n_bytes), raw_dtype=t.tensor_type)
        elif kind in ("hot", "cold"):
            t, l = p[0], p[1]
            _, rows, n_per_row = reversed([int(x) for x in t.shape])
            sel = hot[l] if kind == "hot" else cold[l]
            qt = t.tensor_type if (kind == "hot" or cold_qt[p[2]] is None) else cold_qt[p[2]]
            rb = row_bytes(qt, n_per_row)
            n = len(sel) + n_zero
            if qt in (QT.F32, QT.F16):
                w.add_tensor_info(name, [n, rows, n_per_row], np.float32 if qt == QT.F32 else np.float16,
                                  n * rows * rb, raw_dtype=qt)
            else:
                w.add_tensor_info(name, [n, rows, rb], np.uint8, n * rows * rb, raw_dtype=qt)
        else:
            a = id_maps(n_expert, hot[p], cold[p])[kind]
            w.add_tensor_info(name, list(a.shape), a.dtype, a.nbytes)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()

    procs, pool, ck, imx = [], None, None, None
    if any(v is not None for v in cold_qt.values()):
        from smel.gguf_patch import GGMLProc, Imatrix
        from smel.model_io import Checkpoint
        if not hf or not imatrix:
            raise ValueError("hf and imatrix are needed to re-quantize the cold experts")
        ck, imx = Checkpoint(hf), Imatrix(imatrix)
        procs = [GGMLProc(llama_cpp) for _ in range(workers)]
        pool = ThreadPoolExecutor(workers)

    def cold_layer(l):
        """the cold experts of layer l re-quantized: {projection: bytes}"""
        gu = ck.get_rows(ck.full_name(l, "mlp.experts.gate_up_proj"), cold[l])   # [n_cold, 2I, H], gate rows first
        dn = ck.get_rows(ck.full_name(l, "mlp.experts.down_proj"), cold[l])      # [n_cold, H, I]
        inter = dn.shape[2]
        mats = {"gate": gu[:, :inter, :], "up": gu[:, inter:, :], "down": dn}
        mats = {k: m for k, m in mats.items() if cold_qt[k] is not None}
        n_chunks = max(1, len(procs) // len(mats))   # ggml_quantize_chunk is single-threaded: one process per chunk
        jobs, i = {}, 0
        for kind, m in mats.items():
            qt = cold_qt[kind]
            im = imx.get(f"blk.{l}.ffn_{kind}_exps.weight")
            if im is None and qt in (QT.IQ2_XXS, QT.IQ2_XS, QT.IQ1_S, QT.IQ1_M, QT.IQ3_XXS):
                raise RuntimeError(f"{qt.name} needs an importance matrix: none for blk.{l}.ffn_{kind}_exps")
            im = None if im is None else im[cold[l]]
            m = m.float().numpy()
            for c in np.array_split(np.arange(len(m)), n_chunks):
                jobs[(kind, int(c[0]))] = pool.submit(procs[i % len(procs)].quantize, m[c], int(qt),
                                                      None if im is None else im[c])
                i += 1
        return {k: b"".join(jobs[j].result() for j in sorted(j for j in jobs if j[0] == k)) for k in mats}

    t0, cur_l, cur_cold, done = time.time(), None, None, 0
    for name, kind, p in plan:
        if kind == "copy":
            w.write_tensor_data(raw(p))
        elif kind == "hot":
            t, l = p
            e = raw(t).reshape(int(t.shape[-1]), -1)
            w.write_tensor_data(np.concatenate([e[hot[l]].reshape(-1), np.zeros(n_zero * e.shape[1], np.uint8)]))
        elif kind == "cold":
            t, l, k = p
            if cold_qt[k] is None:
                e = raw(t).reshape(int(t.shape[-1]), -1)
                w.write_tensor_data(np.concatenate([e[cold[l]].reshape(-1), np.zeros(n_zero * e.shape[1], np.uint8)]))
            else:
                if cur_l != l:
                    cur_l, cur_cold = l, cold_layer(l)
                q = np.frombuffer(cur_cold[k], dtype=np.uint8)
                w.write_tensor_data(np.concatenate([q, np.zeros(n_zero * (len(q) // len(cold[l])), np.uint8)]))
            if k == "down":
                done += 1
                el = time.time() - t0
                log(f"layer {l}: {len(hot[l])} hot / {len(cold[l])} cold, {done}/{len(layers)} layers, "
                    f"{el:.0f}s, ETA {el / done * (len(layers) - done):.0f}s")
        else:
            w.write_tensor_data(id_maps(n_expert, hot[p], cold[p])[kind])
    w.close()
    for pr in procs:
        pr.close()
    return out_path
