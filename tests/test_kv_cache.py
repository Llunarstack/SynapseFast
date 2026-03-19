import torch

import synapsefast as sf
from synapsefast._torch_reference import kv_cache_prefill_torch, kv_cache_decode_torch


def test_kv_cache_prefill_cpu_matches_reference():
    torch.manual_seed(0)
    B, H, L, T, D = 2, 3, 16, 5, 32
    start_pos = 7

    k_cache = torch.zeros(B, H, L, D, dtype=torch.float32)
    v_cache = torch.zeros(B, H, L, D, dtype=torch.float32)
    k_new = torch.randn(B, H, T, D, dtype=torch.float32)
    v_new = torch.randn(B, H, T, D, dtype=torch.float32)

    sf.kv_cache_prefill(k_cache, v_cache, k_new, v_new, start_pos=start_pos)

    k_ref = torch.zeros_like(k_cache)
    v_ref = torch.zeros_like(v_cache)
    kv_cache_prefill_torch(k_ref, v_ref, k_new, v_new, start_pos=start_pos)

    assert torch.allclose(k_cache, k_ref, atol=0.0, rtol=0.0)
    assert torch.allclose(v_cache, v_ref, atol=0.0, rtol=0.0)


def test_kv_cache_decode_cpu_matches_reference():
    torch.manual_seed(0)
    B, H, L, D = 1, 2, 24, 32
    pos = 16

    k_cache = torch.randn(B, H, L, D, dtype=torch.float32)
    v_cache = torch.randn(B, H, L, D, dtype=torch.float32)
    q = torch.randn(B, H, 1, D, dtype=torch.float32)

    out_sf = sf.kv_cache_decode(q, k_cache, v_cache, pos=pos, causal=True)
    out_ref = kv_cache_decode_torch(q, k_cache, v_cache, pos=pos, causal=True)
    assert torch.allclose(out_sf, out_ref, atol=0.0, rtol=0.0)
