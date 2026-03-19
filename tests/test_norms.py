import torch

import synapsefast as sf
from synapsefast._torch_reference import rms_norm_torch, layer_norm_torch


def test_rms_norm_matches_torch():
    torch.manual_seed(0)
    B, H, T, D = 2, 3, 5, 64
    x = torch.randn(B, H, T, D, dtype=torch.float32)
    weight = torch.randn(D, dtype=torch.float32)

    out_sf = sf.rms_norm(x, weight, eps=1e-5)
    out_ref = rms_norm_torch(x, weight, eps=1e-5)
    assert torch.allclose(out_sf, out_ref, atol=0.0, rtol=0.0)


def test_layer_norm_matches_torch():
    torch.manual_seed(0)
    B, H, T, D = 2, 3, 5, 64
    x = torch.randn(B, H, T, D, dtype=torch.float32)
    weight = torch.randn(D, dtype=torch.float32)
    bias = torch.randn(D, dtype=torch.float32)

    out_sf = sf.layer_norm(x, weight, bias, eps=1e-5)
    out_ref = layer_norm_torch(x, weight, bias, eps=1e-5)
    assert torch.allclose(out_sf, out_ref, atol=0.0, rtol=0.0)
