"""Layer weight sources with double-buffered prefetch: the next layer is read (mmap / pinned RAM) and copied to the
GPU on a side CUDA stream in a worker thread while the current layer computes."""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from safetensors import safe_open

from .model_io import Checkpoint, Dims
from .moe import BF16Experts, Int4Experts, LayerExperts
from .layer_runner import LayerWeights


def build_layer(idx, ltype, t: dict, sel_ids=None, group=32, frozen_dtype=torch.bfloat16, act="silu"):
    """t: device tensors of one layer, names relative to the layer. Recognizes either bf16 fused experts
    ('mlp.experts.gate_up_proj'/'down_proj') or packed int4 ('experts.gate_up.q', ...) + 'experts.sel.*'."""
    dense = {k: v for k, v in t.items() if not k.startswith(("mlp.experts.", "experts."))}
    if "mlp.experts.gate_up_proj" in t:
        gu, dn = t["mlp.experts.gate_up_proj"], t["mlp.experts.down_proj"]
        frozen = BF16Experts(gu, dn)
        sgu = sdn = None
        if sel_ids is not None and len(sel_ids):
            ids = torch.as_tensor(sel_ids, device=gu.device)
            sgu, sdn = gu[ids].clone(), dn[ids].clone()
    else:
        gq = {k[len("experts.gate_up."):]: v for k, v in t.items() if k.startswith("experts.gate_up.")}
        dq = {k[len("experts.down."):]: v for k, v in t.items() if k.startswith("experts.down.")}
        frozen = Int4Experts(gq, dq, dtype=frozen_dtype)
        sgu, sdn = t.get("experts.sel.gate_up"), t.get("experts.sel.down")
        if sel_ids is None and "experts.sel.ids" in t:
            sel_ids = t["experts.sel.ids"].tolist()
    if sgu is None:
        sel_ids = None
    return LayerWeights(idx, ltype, dense, LayerExperts(frozen, sel_ids, sgu, sdn, group=group, act=act))


class _StreamingSource:
    """Base class: subclasses implement _cpu(l) -> dict of CPU tensors and _make(l, gpu_dict) -> LayerWeights."""

    def __init__(self, device="cuda", resident=False, max_on_gpu=2):
        self.device = torch.device(device)
        self.resident = resident
        self.max_on_gpu = max_on_gpu
        self.cache = {}  # l -> LayerWeights on device
        self.pending = {}  # l -> Future[(LayerWeights, event)]
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.side = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None
        self.stats = {"load_s": 0.0, "wait_s": 0.0, "bytes": 0, "loads": 0}
        self._lock = threading.Lock()

    def _upload(self, l):
        t0 = time.perf_counter()
        cpu = self._cpu(l)
        nb = sum(v.numel() * v.element_size() for v in cpu.values())
        if self.side is not None:
            with torch.cuda.stream(self.side):
                gpu = {k: v.to(self.device, non_blocking=True) for k, v in cpu.items()}
                ev = torch.cuda.Event()
                ev.record(self.side)
            ev.synchronize()  # keep CPU buffers alive until the copy finished (worker thread only)
        else:
            gpu, ev = {k: v.to(self.device) for k, v in cpu.items()}, None
        with self._lock:
            self.stats["load_s"] += time.perf_counter() - t0
            self.stats["bytes"] += nb
            self.stats["loads"] += 1
        return gpu

    def _finish(self, l, gpu):
        if self.side is not None:
            cur = torch.cuda.current_stream(self.device)
            for v in gpu.values():
                v.record_stream(cur)
        LW = self._make(l, gpu)
        self.cache[l] = LW
        self._evict(keep=l)
        return LW

    def _evict(self, keep):
        if self.resident:
            return
        while len(self.cache) > self.max_on_gpu - len(self.pending):
            victims = [k for k in self.cache if k != keep]
            if not victims:
                break
            del self.cache[victims[0]]

    def prefetch(self, l):
        if l in self.cache or l in self.pending:
            return
        if not self.resident:
            # make room before the copy starts
            while len(self.cache) >= self.max_on_gpu:
                victims = list(self.cache)
                del self.cache[victims[0]]
        self.pending[l] = self.pool.submit(self._upload, l)

    def get(self, l):
        if l in self.cache:
            # move to most-recent position
            LW = self.cache.pop(l)
            self.cache[l] = LW
            return LW
        t0 = time.perf_counter()
        fut = self.pending.pop(l, None)
        gpu = fut.result() if fut is not None else self._upload(l)
        self.stats["wait_s"] += time.perf_counter() - t0
        return self._finish(l, gpu)

    def close(self):
        self.pool.shutdown(wait=True)
        self.cache.clear()


class CheckpointSource(_StreamingSource):
    """Layers straight from the original safetensors (bf16). Used by probe/cache and tests."""

    def __init__(self, ckpt: Checkpoint, device="cuda", dtype=torch.bfloat16, selection=None, resident=False,
                 group=32, frozen="bf16", quant_group=64):
        super().__init__(device, resident)
        self.ckpt, self.dtype, self.selection = ckpt, dtype, selection or {}
        self.group, self.frozen, self.quant_group = group, frozen, quant_group
        self.d: Dims = ckpt.dims

    def _cpu(self, l):
        t = self.ckpt.layer(l)
        if self.frozen == "int4":
            from .pack import pack_layer
            return pack_layer(t, self.selection.get(l), self.quant_group, self.dtype)
        out = {}
        for k, v in t.items():
            keep_fp32 = k.endswith(("A_log", "dt_bias")) or not v.is_floating_point()
            out[k] = v if keep_fp32 else v.to(self.dtype)
        return out

    def _make(self, l, gpu):
        return build_layer(l, self.d.layer_types[l], gpu, self.selection.get(l), self.group, self.dtype, self.d.act)


class PackedSource(_StreamingSource):
    """Layers from the packed block (one safetensors per layer). With pin=True every layer is read once into pinned
    RAM (chunked per tensor), so a step costs only H2D copies; otherwise it is read from mmap each time."""

    def __init__(self, packed_dir, dims: Dims, layers, device="cuda", resident=False, pin=False, group=32,
                 pin_budget_gb=9.0, preloaded=None, selection=None, ckpt: Checkpoint = None):
        """preloaded: {l: dict of CPU tensors from pack_layer} used instead of files (no packed block on disk).
        selection + ckpt: train a different expert set on the same packed block. The int4 bank holds all experts,
        so only the bf16 copies of the newly selected experts are read (once) from the original checkpoint."""
        super().__init__(device, resident)
        self.dir, self.d, self.group = packed_dir, dims, group
        self.pinned = {}
        self.preloaded = preloaded or {}
        self.pinned_gb = 0.0
        self.sel_rows = None
        if selection is not None:
            self.sel_rows = {}
            for l in layers:
                ids = list(selection.get(l, []))
                if ids:
                    gu_name = ckpt.full_name(l, "mlp.experts.gate_up_proj")
                    dn_name = ckpt.full_name(l, "mlp.experts.down_proj")
                    self.sel_rows[l] = {"experts.sel.ids": torch.tensor(ids, dtype=torch.long),
                                        "experts.sel.gate_up": ckpt.get_rows(gu_name, ids).contiguous(),
                                        "experts.sel.down": ckpt.get_rows(dn_name, ids).contiguous()}
        if pin:
            used = 0
            for l in layers:
                t = self._read(l)
                nb = sum(v.numel() * v.element_size() for v in t.values())
                if (used + nb) / 2**30 > pin_budget_gb:
                    break
                self.pinned[l] = {k: v.pin_memory() for k, v in t.items()}
                self.preloaded.pop(l, None)
                used += nb
            self.pinned_gb = used / 2**30

    def _path(self, l):
        return os.path.join(self.dir, f"layer_{l:02d}.safetensors")

    def _read(self, l):
        if l in self.preloaded:
            t = self.preloaded[l]
        else:
            with safe_open(self._path(l), "pt") as f:
                t = {k: f.get_tensor(k) for k in f.keys()}
        if self.sel_rows is not None:
            t = {k: v for k, v in t.items() if not k.startswith("experts.sel.")}
            t.update(self.sel_rows.get(l, {}))
        return t

    def _cpu(self, l):
        return self.pinned[l] if l in self.pinned else self._read(l)

    def _make(self, l, gpu):
        return build_layer(l, self.d.layer_types[l], gpu, None, self.group, act=self.d.act)


class DictSource:
    """Already materialized LayerWeights (tests / fully resident block)."""

    def __init__(self, layers: dict):
        self.layers = layers
        self.stats = {"load_s": 0.0, "wait_s": 0.0, "bytes": 0, "loads": 0}

    def get(self, l):
        return self.layers[l]

    def prefetch(self, l):
        pass

    def close(self):
        pass
