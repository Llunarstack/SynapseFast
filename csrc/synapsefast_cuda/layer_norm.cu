#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// LayerNorm (mean/variance) over the last dimension for fp16/bf16.

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(scalar_t x) {
  return static_cast<float>(x);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t store_from_float(float x) {
  return static_cast<scalar_t>(x);
}

template <typename scalar_t>
__global__ void layer_norm_fwd_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int N,
    int D,
    float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  __shared__ float sdata[256];

  // Pass 1: mean
  float sum = 0.0f;
  if (tid < D) {
    int off = row * D + tid;
    sum = load_as_float<scalar_t>(x[off]);
  }
  sdata[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) sdata[tid] += sdata[tid + stride];
    __syncthreads();
  }

  float mean = 0.0f;
  if (tid == 0) {
    mean = sdata[0] / (float)D;
    sdata[0] = mean;
  }
  __syncthreads();
  mean = sdata[0];

  // Pass 2: variance
  float var_sum = 0.0f;
  if (tid < D) {
    int off = row * D + tid;
    float xf = load_as_float<scalar_t>(x[off]);
    float d = xf - mean;
    var_sum = d * d;
  }
  sdata[tid] = var_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) sdata[tid] += sdata[tid + stride];
    __syncthreads();
  }

  float inv_std = 0.0f;
  if (tid == 0) {
    float var = sdata[0] / (float)D;
    inv_std = rsqrtf(var + eps);
    sdata[0] = inv_std;
  }
  __syncthreads();
  inv_std = sdata[0];

  if (tid < D) {
    int off = row * D + tid;
    float xf = load_as_float<scalar_t>(x[off]);
    float wf = load_as_float<scalar_t>(weight[tid]);
    float bf = load_as_float<scalar_t>(bias[tid]);
    float y = (xf - mean) * inv_std * wf + bf;
    out[off] = store_from_float<scalar_t>(y);
  }
}

torch::Tensor layer_norm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(weight.is_cuda() && bias.is_cuda(), "weight/bias must be CUDA");
  TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dim");
  TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1, "weight/bias must be [D]");
  TORCH_CHECK(x.size(-1) == weight.size(0) && x.size(-1) == bias.size(0), "Last dim must match weight/bias");

  x = x.contiguous();
  weight = weight.contiguous();
  bias = bias.contiguous();

  int64_t D = x.size(-1);
  int64_t N = x.numel() / D;
  auto out = torch::empty_like(x);

  dim3 grid(N);
  dim3 block(256);
  float eps_f = (float)eps;
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  if (x.scalar_type() == at::kHalf) {
    layer_norm_fwd_kernel<at::Half><<<grid, block, 0, stream>>>(
        reinterpret_cast<at::Half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(out.data_ptr<at::Half>()),
        (int)N, (int)D, eps_f);
  } else if (x.scalar_type() == at::kBFloat16) {
    layer_norm_fwd_kernel<at::BFloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<at::BFloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<at::BFloat16*>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<at::BFloat16*>(bias.data_ptr<at::BFloat16>()),
        reinterpret_cast<at::BFloat16*>(out.data_ptr<at::BFloat16>()),
        (int)N, (int)D, eps_f);
  } else {
    TORCH_CHECK(false, "layer_norm_forward supports fp16/bf16 for this MVP kernel.");
  }

  return out;
}

