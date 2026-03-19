import shutil

import pytest
import torch

import synapsefast as sf
from synapsefast._torch_reference import kv_cache_prefill_torch, kv_cache_decode_torch


def _cuda_ext_available() -> bool:
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
def test_kv_cache_prefill_decode_cuda_matches_reference_fp16():
    torch.manual_seed(0)
    device = "cuda"

    B, H, L, T, D = 2, 2, 32, 6, 64
    start_pos = 10
    pos = 13

    k_cache = torch.zeros(B, H, L, D, device=device, dtype=torch.float16)
    v_cache = torch.zeros(B, H, L, D, device=device, dtype=torch.float16)
    k_new = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    v_new = torch.randn(B, H, T, D, device=device, dtype=torch.float16)

    # Prefill (in-place)
    sf.kv_cache_prefill(k_cache, v_cache, k_new, v_new, start_pos=start_pos)

    # Prefill reference
    k_ref = torch.zeros_like(k_cache)
    v_ref = torch.zeros_like(v_cache)
    kv_cache_prefill_torch(k_ref, v_ref, k_new, v_new, start_pos=start_pos)
    assert torch.allclose(k_cache, k_ref, atol=0.0, rtol=0.0)
    assert torch.allclose(v_cache, v_ref, atol=0.0, rtol=0.0)

    q = torch.randn(B, H, 1, D, device=device, dtype=torch.float16)
    out_sf = sf.kv_cache_decode(q, k_cache, v_cache, pos=pos, causal=True)
    out_ref = kv_cache_decode_torch(q, k_ref, v_ref, pos=pos, causal=True)

    assert torch.allclose(out_sf, out_ref, atol=5e-2, rtol=5e-2)
