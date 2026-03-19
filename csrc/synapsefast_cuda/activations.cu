#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(scalar_t x) {
  return static_cast<float>(x);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t store_from_float(float x) {
  return static_cast<scalar_t>(x);
}

template <typename scalar_t>
__global__ void gelu_fwd_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ out, int64_t n) {
  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= n) return;

  float xf = load_as_float<scalar_t>(x[idx]);
  // GELU (tanh approximation) to match torch approximate="tanh".
  float c = 0.7978845608028654f; // sqrt(2/pi)
  float x3 = xf * xf * xf;
  float inner = c * (xf + 0.044715f * x3);
  float y = 0.5f * xf * (1.0f + tanhf(inner));
  out[idx] = store_from_float<scalar_t>(y);
}

torch::Tensor gelu_forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "gelu_forward expects CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16, "gelu_forward supports fp16/bf16");

  x = x.contiguous();
  auto out = torch::empty_like(x);
  int64_t n = x.numel();

  const int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  if (x.scalar_type() == at::kHalf) {
    gelu_fwd_kernel<at::Half><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<at::Half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(out.data_ptr<at::Half>()),
        n);
  } else if (x.scalar_type() == at::kBFloat16) {
    gelu_fwd_kernel<at::BFloat16><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<at::BFloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<at::BFloat16*>(out.data_ptr<at::BFloat16>()),
        n);
  }
  return out;
}

