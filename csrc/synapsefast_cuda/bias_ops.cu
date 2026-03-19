#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void bias_add_kernel(const scalar_t* __restrict__ x, const scalar_t* __restrict__ bias, scalar_t* __restrict__ out, int64_t n, int64_t D) {
  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= n) return;
  int64_t b = idx % D;
  out[idx] = x[idx] + bias[b];
}

torch::Tensor bias_add_forward(torch::Tensor x, torch::Tensor bias) {
  TORCH_CHECK(x.is_cuda(), "bias_add_forward expects CUDA x");
  TORCH_CHECK(bias.is_cuda(), "bias_add_forward expects CUDA bias");
  TORCH_CHECK(x.scalar_type() == bias.scalar_type(), "x and bias must have same dtype");
  TORCH_CHECK(bias.dim() == 1, "bias must be 1D [D]");
  TORCH_CHECK(x.size(-1) == bias.size(0), "x last dim must match bias length");

  x = x.contiguous();
  bias = bias.contiguous();

  auto out = torch::empty_like(x);
  int64_t n = x.numel();
  int64_t D = bias.size(0);

  int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  if (x.scalar_type() == at::kHalf) {
    bias_add_kernel<at::Half><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<at::Half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(out.data_ptr<at::Half>()),
        n, D);
  } else if (x.scalar_type() == at::kBFloat16) {
    bias_add_kernel<at::BFloat16><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<at::BFloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<at::BFloat16*>(bias.data_ptr<at::BFloat16>()),
        reinterpret_cast<at::BFloat16*>(out.data_ptr<at::BFloat16>()),
        n, D);
  } else {
    TORCH_CHECK(false, "bias_add_forward supports fp16/bf16 only for this MVP.");
  }

  return out;
}

