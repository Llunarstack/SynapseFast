import torch

import synapsefast as sf
from synapsefast._torch_reference import attention_torch


def test_attention_cpu_fp16_matches_torch():
    torch.manual_seed(0)
    B, H, T, D = 2, 4, 16, 32
    q = torch.randn(B, H, T, D, dtype=torch.float16)
    k = torch.randn(B, H, T, D, dtype=torch.float16)
    v = torch.randn(B, H, T, D, dtype=torch.float16)

    out_sf = sf.attention(q, k, v, causal=False)
    out_ref = attention_torch(q, k, v, causal=False)
    assert torch.allclose(out_sf, out_ref, atol=1e-3, rtol=1e-3)


@torch.inference_mode()
def test_attention_causal_cpu():
    torch.manual_seed(0)
    B, H, T, D = 1, 2, 17, 32
    q = torch.randn(B, H, T, D, dtype=torch.float32)
    k = torch.randn(B, H, T, D, dtype=torch.float32)
    v = torch.randn(B, H, T, D, dtype=torch.float32)

    out_sf = sf.attention(q, k, v, causal=True)
    out_ref = attention_torch(q, k, v, causal=True)
    assert torch.allclose(out_sf, out_ref, atol=1e-6, rtol=1e-6)
