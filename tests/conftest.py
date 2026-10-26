import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smel.tiny import build_tiny_model, save_tiny_checkpoint  # noqa: E402

DEV = "cuda"
# exact fp32 references: HF's conv1d would otherwise run in TF32 through cuDNN
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


@pytest.fixture(scope="session")
def model32():
    return build_tiny_model(seed=0, dtype=torch.float32, device=DEV)


@pytest.fixture(scope="session")
def model16():
    return build_tiny_model(seed=0, dtype=torch.bfloat16, device=DEV)


@pytest.fixture(scope="session")
def tiny_ckpt(tmp_path_factory, model16):
    return save_tiny_checkpoint(model16, str(tmp_path_factory.mktemp("tiny_ckpt")))


@pytest.fixture(scope="session")
def seqs():
    """Three sequences of different lengths with supervised targets on part of each."""
    g = torch.Generator().manual_seed(123)
    out = []
    for n in (37, 70, 129):
        ids = torch.randint(0, 512, (n,), generator=g)
        tgt = torch.full((n,), -100, dtype=torch.long)
        a = n // 3
        tgt[a:n - 1] = ids[a + 1:n]
        out.append((ids.to(DEV), tgt.to(DEV)))
    return out


@pytest.fixture(scope="session")
def tokenizer():
    """Real Qwen3.6 tokenizer + chat template: env SMEL_TOKENIZER (a local dir) or a download of the small
    tokenizer files from the Hugging Face Hub (no weights)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.environ.get("SMEL_TOKENIZER", "Qwen/Qwen3.6-35B-A3B"))
