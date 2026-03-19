#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(scalar_t x) {
  return static_cast<float>(x);
}

template <>
__device__ __forceinline__ float load_as_float<at::Half>(at::Half x) {
  return __half2float((__half)x);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t store_from_float(float x) {
  return static_cast<scalar_t>(x);
}

template <>
__device__ __forceinline__ at::Half store_from_float<at::Half>(float x) {
  return (at::Half)__float2half(x);
}

// x: [N, D], weight: [D]
template <typename scalar_t>
__global__ void rms_norm_fwd_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ out,
    int N,
    int D,
    float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  __shared__ float sdata[256];

  float sum_sq = 0.0f;
  if (tid < D) {
    int off = row * D + tid;
    float xf = load_as_float<scalar_t>(x[off]);
    sum_sq = xf * xf;
  }
  sdata[tid] = sum_sq;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) sdata[tid] += sdata[tid + stride];
    __syncthreads();
  }

  float inv_rms = 0.0f;
  if (tid == 0) {
    float mean_sq = sdata[0] / (float)D;
    inv_rms = rsqrtf(mean_sq + eps);
    sdata[0] = inv_rms;
  }
  __syncthreads();

  float scale = sdata[0];
  if (tid < D) {
    int off = row * D + tid;
    float xf = load_as_float<scalar_t>(x[off]);
    float wf = load_as_float<scalar_t>(weight[tid]);
    out[off] = store_from_float<scalar_t>(xf * scale * wf);
  }
}

torch::Tensor rms_norm_forward(torch::Tensor x, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dim");
  TORCH_CHECK(weight.dim() == 1, "weight must be [D]");
  TORCH_CHECK(x.size(-1) == weight.size(0), "x last dim must match weight");

  x = x.contiguous();
  weight = weight.contiguous();

  int64_t D = x.size(-1);
  int64_t N = x.numel() / D;
  auto out = torch::empty_like(x);

  int threads = 256;
  dim3 grid(N);
  dim3 block(threads);
  float eps_f = (float)eps;

  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  if (x.scalar_type() == at::kHalf) {
    rms_norm_fwd_kernel<at::Half><<<grid, block, 0, stream>>>(
        reinterpret_cast<at::Half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(out.data_ptr<at::Half>()),
        (int)N, (int)D, eps_f);
  } else if (x.scalar_type() == at::kBFloat16) {
    rms_norm_fwd_kernel<at::BFloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<at::BFloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<at::BFloat16*>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<at::BFloat16*>(out.data_ptr<at::BFloat16>()),
        (int)N, (int)D, eps_f);
  } else {
    TORCH_CHECK(false, "rms_norm_forward supports fp16/bf16 for this MVP kernel.");
  }

  return out;
}

