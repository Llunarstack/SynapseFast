# SynapseFast

An experimental ML ops library with a clean Python API and a high-performance architecture:
- Rust for planning/dispatch (autotuning heuristics, backend selection)
- C++/CUDA for optional GPU kernels
- Zig build scripts for reproducible kernel/build orchestration (optional)

This repo is correctness-first. When CUDA kernels are unavailable, the Python API falls back to PyTorch reference implementations.

## Install

```bash
pip install -e .
```

Optional CUDA kernels compile at runtime (when `nvcc` is available). If you want to prebuild kernels, see `zig/build.zig`.

On Windows, CUDA extension builds require MSVC (`cl.exe`). If you don't want to manually open a VS Developer Prompt, use:

```powershell
.\scripts\run_with_msvc.ps1 python -c "import synapsefast._cuda_ops as co; print(co.cuda_ext_loaded())"
```

## Usage

```python
import torch
import synapsefast as sf

B, H, T, D = 2, 8, 128, 64
q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)

out = sf.attention(q, k, v, causal=False)
```

Optional backend control:

```python
out = sf.attention(q, k, v, causal=False, backend="torch")  # force PyTorch SDPA
out = sf.attention(q, k, v, causal=False, backend="cuda")   # try custom CUDA kernels
```

## Testing

```bash
pytest -q
```

Tests that require CUDA kernels are skipped when kernels are not available.

## Benchmarking

```bash
python bench/bench_attention.py --dtype fp16 --B 1 --H 8 --D 64 --T_list 128,256,512
```

Add `--causal` for causal attention. If `xformers` is installed, it will be benchmarked as well.

Compare against other attention backends (CUDA events timing):

```bash
python bench/compare_attention.py --dtype fp16 --B 1 --H 8 --D 64 --T_list 128,256,512,1024
python bench/compare_attention.py --dtype fp16 --causal --B 1 --H 8 --D 64 --T_list 128,256,512,1024
```

Latest benchmark plots:
- `docs/benchmarks/compare_noncausal.png`
- `docs/benchmarks/compare_causal.png`

Max-speed mode (autotunes + caches best backend per shape/GPU):

```powershell
$env:SYNAPSEFAST_AUTOTUNE="1"
python bench/autotune_attention.py --dtype fp16 --B 1 --H 8 --D 64 --T_list 128,256,512,1024
```

Note on `backend="auto"`:
- `backend="auto"` defaults to **PyTorch SDPA** unless autotune is enabled.
- To force SynapseFast custom CUDA attention, use `backend="cuda"` or:

```powershell
$env:SYNAPSEFAST_FORCE_CUSTOM_CUDA="1"
```

## Training Example (Toy GPT)

This repo includes a tiny GPT-style training script that uses `synapsefast` attention/norm/gelu.

```bat
set SYNAPSEFAST_USE_CUSTOM_CUDA=1
python examples/train_toy_gpt.py --device cuda --dtype fp16 --seq_len 128 --batch_size 4 --steps 50
```

Resume from a checkpoint:

```bat
python examples/train_toy_gpt.py --device cuda --dtype fp16 --resume runs\<your_run>\ckpt_step_00000050.pt
```

## “Classic ML” (NumPy / Pandas / scikit-learn / XGBoost)

Install extras:

```bash
pip install -e ".[data,sklearn]"
pip install -e ".[xgb]"
```

Run demos:

```bash
python examples/sklearn_tabular_demo.py
python examples/xgboost_tabular_demo.py
python examples/synapsefast_sklearn_like_demo.py
```

## Optional Integrations (wrappers)

Install extras:

```bash
pip install -e ".[lightgbm]"
pip install -e ".[catboost]"
pip install -e ".[nlp]"
pip install -e ".[spacy]"
pip install -e ".[cv]"
```

Run demos:

```bash
python examples/hf_text_classify_demo.py
```

