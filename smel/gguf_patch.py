"""Patch an existing GGUF in place of a copy: re-quantize only the tensors changed by the merged LoRA, with the same
ggml type per tensor and the same importance matrix, using ggml_quantize_chunk from a llama.cpp build.
Everything else (other layers, MTP head, tokenizer, metadata) stays byte-identical to the original file."""
import ctypes
import os
import sys
import tempfile

import numpy as np
from gguf import GGUFReader

_GGML_LIB = {"win32": "ggml-base.dll", "darwin": "libggml-base.dylib"}.get(sys.platform, "libggml-base.so")


def llama_dir(d=None):
    """llama.cpp build directory: argument, env SMEL_LLAMA_CPP_DIR, or the directory of llama-server on PATH."""
    import shutil
    d = d or os.environ.get("SMEL_LLAMA_CPP_DIR")
    if not d:
        exe = shutil.which("llama-server")
        d = os.path.dirname(exe) if exe else None
    if not d or not os.path.exists(os.path.join(d, _GGML_LIB)):
        raise SystemExit(f"llama.cpp build with {_GGML_LIB} not found: set SMEL_LLAMA_CPP_DIR or llamacpp.dir in "
                         f"the config (got {d!r})")
    return d


class GGML:
    def __init__(self, path=None):
        path = llama_dir(path)
        if sys.platform == "win32":
            os.add_dll_directory(path)
        self.lib = ctypes.CDLL(os.path.join(path, _GGML_LIB))
        L = self.lib
        L.ggml_quantize_init.argtypes = [ctypes.c_int]
        L.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                          ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        L.ggml_quantize_chunk.restype = ctypes.c_size_t
        L.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        L.ggml_row_size.restype = ctypes.c_size_t
        L.ggml_validate_row_data.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        L.ggml_validate_row_data.restype = ctypes.c_bool

    def quantize(self, w: np.ndarray, qtype: int, imatrix: np.ndarray | None) -> bytes:
        """w: float32 [n_mat, rows, n_per_row] (n_mat = experts, 1 for 2D); imatrix: [n_mat, n_per_row] or None."""
        w = np.ascontiguousarray(w, dtype=np.float32)
        n_mat, rows, n_per_row = w.shape
        self.lib.ggml_quantize_init(qtype)
        row_size = self.lib.ggml_row_size(qtype, n_per_row)
        out = np.empty(n_mat * rows * row_size, dtype=np.uint8)
        im = None if imatrix is None else np.ascontiguousarray(imatrix, dtype=np.float32)
        for e in range(n_mat):
            dst = out[e * rows * row_size:(e + 1) * rows * row_size]
            imp = None if im is None else im[e].ctypes.data
            n = self.lib.ggml_quantize_chunk(qtype, w[e].ctypes.data, dst.ctypes.data, 0, rows, n_per_row, imp)
            assert n == rows * row_size, (n, rows * row_size)
            assert self.lib.ggml_validate_row_data(qtype, dst.ctypes.data, n)
        return out.tobytes()


class GGMLProc:
    """GGML quantization in a separate process: ggml links libomp, torch links libiomp5md, and both in one process
    abort with 'OMP: Error #15'. Same interface as GGML.quantize; data goes through temporary .npy files."""

    def __init__(self, path=None, tmp_dir=None):
        import subprocess
        self.tmp = tmp_dir or tempfile.mkdtemp(prefix="smel_ggml_")
        env = {**os.environ, "SMEL_LLAMA_CPP_DIR": llama_dir(path)}
        self.p = subprocess.Popen([sys.executable, "-m", "smel.gguf_patch"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, env=env,
                                  cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def quantize(self, w, qtype, imatrix):
        import json
        wp, ip, op = (os.path.join(self.tmp, f"q_{k}.npy") for k in ("w", "im", "out"))
        np.save(wp, np.ascontiguousarray(w, dtype=np.float32))
        if imatrix is not None:
            np.save(ip, np.ascontiguousarray(imatrix, dtype=np.float32))
        self.p.stdin.write(json.dumps({"w": wp, "im": ip if imatrix is not None else None, "out": op,
                                       "qtype": int(qtype)}) + "\n")
        self.p.stdin.flush()
        resp = self.p.stdout.readline().strip()
        if resp != "ok":
            raise RuntimeError(f"ggml worker failed: {resp!r}")
        return np.load(op).tobytes()

    def close(self):
        self.p.stdin.close()
        self.p.wait(30)


def _worker():
    import json
    g = GGML()
    for line in sys.stdin:
        try:
            r = json.loads(line)
            im = np.load(r["im"]) if r["im"] else None
            q = g.quantize(np.load(r["w"]), r["qtype"], im)
            np.save(r["out"], np.frombuffer(q, dtype=np.uint8))
            print("ok", flush=True)
        except Exception as e:  # report and keep serving
            print(f"error {type(e).__name__}: {e}", flush=True)


class Imatrix:
    def __init__(self, path):
        self.r = GGUFReader(path)
        self.t = {t.name: t for t in self.r.tensors}

    def get(self, name):
        s, c = self.t.get(name + ".in_sum2"), self.t.get(name + ".counts")
        if s is None:
            return None
        sums = np.array(s.data, dtype=np.float32).reshape(len(np.array(c.data).reshape(-1)), -1)
        counts = np.array(c.data, dtype=np.float32).reshape(-1)
        vals = np.where(counts[:, None] > 0, sums / np.maximum(counts[:, None], 1e-30), 1.0)
        return vals.astype(np.float32)


def hf_to_gguf_tensors(l, gate_up=None, down=None, shared=None, arch="qwen3_5_moe", fused_gate_up=False):
    """Map merged HF tensors of layer l to GGUF names -> float32 [n_mat, rows, n_per_row].

    Experts: ffn_{gate,up,down}_exps, or ffn_gate_up_exps when the GGUF keeps gate and up fused (HF order: gate
    rows first). The MLP trained next to them: Qwen's shared expert (ffn_*_shexp) or Gemma 4's dense MLP (ffn_*)."""
    out = {}
    if gate_up is not None:
        I = down.shape[2]
        gu = gate_up.float().numpy()
        if fused_gate_up:
            out[f"blk.{l}.ffn_gate_up_exps.weight"] = gu
        else:
            out[f"blk.{l}.ffn_gate_exps.weight"] = gu[:, :I, :]
            out[f"blk.{l}.ffn_up_exps.weight"] = gu[:, I:, :]
        out[f"blk.{l}.ffn_down_exps.weight"] = down.float().numpy()
    if shared is not None:
        g, u, dn = shared
        sfx = "" if arch == "gemma4" else "_shexp"
        out[f"blk.{l}.ffn_gate{sfx}.weight"] = g.float().numpy()[None]
        out[f"blk.{l}.ffn_up{sfx}.weight"] = u.float().numpy()[None]
        out[f"blk.{l}.ffn_down{sfx}.weight"] = dn.float().numpy()[None]
    return out


class GGUFTarget:
    """Tensor info (type, byte offset, byte size) of a GGUF file for in-place byte replacement."""

    def __init__(self, path):
        r = GGUFReader(path)
        self.path = path
        self.info = {t.name: (int(t.tensor_type), int(t.data_offset), int(t.n_bytes), t.tensor_type.name)
                     for t in r.tensors}
        self.reader = r

    def bytes_of(self, name):
        t = next(t for t in self.reader.tensors if t.name == name)
        return np.asarray(t.data).view(np.uint8).tobytes()

    def write(self, name, data: bytes, fh):
        qtype, off, nb, _ = self.info[name]
        assert len(data) == nb, (name, len(data), nb)
        fh.seek(off)
        fh.write(data)


if __name__ == "__main__":
    _worker()
