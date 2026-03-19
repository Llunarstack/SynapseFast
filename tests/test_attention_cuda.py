import shutil

import pytest
import torch

import synapsefast as sf
from synapsefast._torch_reference import attention_torch


def _cuda_ext_available() -> bool:
    # If cl isn't present, our JIT CUDA extension can't build and the API falls back to torch.
    if shutil.which("cl") is None:
        return False
    try:
        import synapsefast._cuda_ops as co

        return co._get_ext() is not None
    except Exception:
        return False


CUDA_EXT_AVAILABLE = torch.cuda.is_available() and _cuda_ext_available()


@pytest.mark.skipif(not CUDA_EXT_AVAILABLE, reason="CUDA extension not available/built")
@torch.inference_mode()
def test_attention_cuda_matches_reference_fp16():
    torch.manual_seed(0)
    B, H, T, D = 1, 2, 32, 64
    device = "cuda"

    q = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.float16)

    out_sf = sf.attention(q, k, v, causal=False)
    out_ref = attention_torch(q, k, v, causal=False)
    assert torch.allclose(out_sf, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not CUDA_EXT_AVAILABLE, reason="CUDA extension not available/built")
@torch.inference_mode()
def test_attention_cuda_matches_reference_causal_fp16():
    torch.manual_seed(0)
    B, H, T, D = 1, 2, 24, 32
    device = "cuda"

    q = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.float16)

    out_sf = sf.attention(q, k, v, causal=True)
    out_ref = attention_torch(q, k, v, causal=True)
    assert torch.allclose(out_sf, out_ref, atol=5e-2, rtol=5e-2)
