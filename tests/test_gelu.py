import torch

import synapsefast as sf
from synapsefast._torch_reference import gelu_torch


def test_gelu_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float32)
    out_sf = sf.gelu(x)
    out_ref = gelu_torch(x)
    assert torch.allclose(out_sf, out_ref, atol=0.0, rtol=0.0)
