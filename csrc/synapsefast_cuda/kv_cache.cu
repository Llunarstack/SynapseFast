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

template <>
__device__ __forceinline__ float load_as_float<__half>(__half x) {
  return __half2float(x);
}

template <>
__device__ __forceinline__ float load_as_float<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t store_from_float(float x) {
  return static_cast<scalar_t>(x);
}

template <>
__device__ __forceinline__ __half store_from_float<__half>(float x) {
  return __float2half(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 store_from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}

// Prefill is an in-place copy into the cache.
template <typename scalar_t>
__global__ void kv_prefill_copy_kernel(
    const scalar_t* __restrict__ k_new,
    const scalar_t* __restrict__ v_new,
    scalar_t* __restrict__ k_cache,
    scalar_t* __restrict__ v_cache,
    int B,
    int H,
    int L,
    int Tnew,
    int D,
    int start_pos) {
  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int total = B * H * Tnew * D;
  if (idx >= total) return;

  int d = idx % D;
  int tmp = idx / D;
  int t_rel = tmp % Tnew;
  int bh = tmp / Tnew;

  int b = bh / H;
  int h = bh % H;

  int cache_t = start_pos + t_rel;

  int k_new_off = ((b * H + h) * Tnew + t_rel) * D + d;
  int k_cache_off = ((b * H + h) * L + cache_t) * D + d;

  int v_cache_off = k_cache_off;
  int v_new_off = k_new_off;

  k_cache[k_cache_off] = k_new[k_new_off];
  v_cache[v_cache_off] = v_new[v_new_off];
}

torch::Tensor kv_cache_prefill_forward(
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor k_new,
    torch::Tensor v_new,
    int start_pos) {
  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda() && k_new.is_cuda() && v_new.is_cuda(), "kv_cache_prefill_forward expects CUDA tensors");
  TORCH_CHECK(k_cache.scalar_type() == k_new.scalar_type() && v_cache.scalar_type() == v_new.scalar_type(), "dtypes must match for k/v tensors");
  TORCH_CHECK(k_cache.scalar_type() == v_cache.scalar_type(), "k/v cache dtypes must match");
  TORCH_CHECK(k_cache.dim() == 4 && k_new.dim() == 4, "expected k_cache/v_cache and k_new/v_new to be rank-4");

  int B = (int)k_cache.size(0);
  int H = (int)k_cache.size(1);
  int L = (int)k_cache.size(2);
  int D = (int)k_cache.size(3);
  int Tnew = (int)k_new.size(2);

  TORCH_CHECK(start_pos >= 0 && start_pos + Tnew <= L, "invalid start_pos for kv prefill");
  TORCH_CHECK(k_new.size(0) == B && k_new.size(1) == H && k_new.size(3) == D, "k_new shape mismatch");

  k_cache = k_cache.contiguous();
  v_cache = v_cache.contiguous();
  k_new = k_new.contiguous();
  v_new = v_new.contiguous();

  int total = B * H * Tnew * D;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  if (k_cache.scalar_type() == at::kHalf) {
    kv_prefill_copy_kernel<at::Half><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<at::Half*>(k_new.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(v_new.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(v_cache.data_ptr<at::Half>()),
        B, H, L, Tnew, D, start_pos);
  } else if (k_cache.scalar_type() == at::kBFloat16) {
    kv_prefill_copy_kernel<at::BFloat16><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<at::BFloat16*>(k_new.data_ptr<at::BFloat16>()),
        reinterpret_cast<at::BFloat16*>(v_new.data_ptr<at::BFloat16>()),
        reinterpret_cast<at::BFloat16*>(k_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<at::BFloat16*>(v_cache.data_ptr<at::BFloat16>()),
        B, H, L, Tnew, D, start_pos);
  } else {
    TORCH_CHECK(false, "kv_cache_prefill_forward supports fp16/bf16 only for this MVP.");
  }

  return k_cache;
}

// Decode: q is a single token, attend over keys up to `pos` (inclusive).
template <typename scalar_t>
__global__ void kv_decode_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_cache,
    const scalar_t* __restrict__ v_cache,
    scalar_t* __restrict__ out,
    int L,
    int D,
    int pos,
    float scale,
    bool causal) {
  // One block per (batch, head).
  int bh = blockIdx.x;
  int tid = threadIdx.x;

  // Effective range: causal attends up to pos, non-causal attends full L.
  int t_end = causal ? pos : (L - 1);
  t_end = t_end < 0 ? -1 : t_end;

  __shared__ float sdata[128];
  __shared__ float m_s;
  __shared__ float l_s;
  __shared__ float scale_s;
  __shared__ float exp_s;
  __shared__ float inv_l_s;

  float q_val = 0.0f;
  if (tid < D) {
    // q layout: [B,H,1,D] -> flatten [B*H*D] since seqlen=1
    int q_off = bh * D + tid;
    q_val = load_as_float<scalar_t>(q[q_off]);
  }

  float o_acc = 0.0f;

  if (tid == 0) {
    m_s = -INFINITY;
    l_s = 0.0f;
  }
  __syncthreads();

  for (int t = 0; t <= t_end; ++t) {
    float partial = 0.0f;
    if (tid < D) {
      int k_off = (bh * L + t) * D + tid;
      float k_val = load_as_float<scalar_t>(k_cache[k_off]);
      partial = q_val * k_val;
    }

    sdata[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) sdata[tid] += sdata[tid + stride];
      __syncthreads();
    }

    if (tid == 0) {
      // For decode, t is always within range, so no additional mask needed.
      float score = sdata[0] * scale;
      float m_old = m_s;
      float m_new = fmaxf(m_old, score);

      float s = 1.0f;
      if (m_new != m_old) s = expf(m_old - m_new);

      scale_s = s;
      m_s = m_new;

      float l_old = l_s;
      l_s = l_old * scale_s;

      exp_s = expf(score - m_new);
      l_s = l_s + exp_s;
    }

    __syncthreads();

    if (tid < D) {
      if (scale_s != 1.0f) o_acc = o_acc * scale_s;
      if (exp_s != 0.0f) {
        int v_off = (bh * L + t) * D + tid;
        float v_val = load_as_float<scalar_t>(v_cache[v_off]);
        o_acc = o_acc + exp_s * v_val;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    inv_l_s = 1.0f / (l_s + 1e-6f);
  }
  __syncthreads();

  if (tid < D) {
    int out_off = bh * D + tid;
    float y = o_acc * inv_l_s;
    out[out_off] = store_from_float<scalar_t>(y);
  }
}

torch::Tensor kv_cache_decode_forward(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    int pos,
    bool causal) {
  TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(), "kv_cache_decode_forward expects CUDA tensors");
  TORCH_CHECK(q.scalar_type() == k_cache.scalar_type() && q.scalar_type() == v_cache.scalar_type(), "q/k/v cache dtypes must match");
  TORCH_CHECK(q.dim() == 4 && k_cache.dim() == 4, "expected rank-4 tensors");
  TORCH_CHECK(q.size(2) == 1, "q must be [B,H,1,D] for decode");
  TORCH_CHECK(k_cache.size(0) == q.size(0) && k_cache.size(1) == q.size(1), "batch/head mismatch");
  TORCH_CHECK(k_cache.size(3) == q.size(3), "head_dim mismatch");

  int B = (int)q.size(0);
  int H = (int)q.size(1);
  int L = (int)k_cache.size(2);
  int D = (int)q.size(3);

  TORCH_CHECK(D <= 128, "MVP decode supports head_dim <= 128.");
  TORCH_CHECK(pos >= 0 && pos < L, "pos out of range");

  q = q.contiguous();
  k_cache = k_cache.contiguous();
  v_cache = v_cache.contiguous();

  auto out = torch::empty_like(q);
  int blocks = B * H;
  const int threads = 128;
  float scale = 1.0f / sqrtf((float)D);
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  if (q.scalar_type() == at::kHalf) {
    kv_decode_kernel<at::Half><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<at::Half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(v_cache.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(out.data_ptr<at::Half>()),
        L, D, pos, scale, causal);
  } else if (q.scalar_type() == at::kBFloat16) {
    kv_decode_kernel<at::BFloat16><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<at::BFloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<at::BFloat16*>(k_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<at::BFloat16*>(v_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<at::BFloat16*>(out.data_ptr<at::BFloat16>()),
        L, D, pos, scale, causal);
  } else {
    TORCH_CHECK(false, "kv_cache_decode_forward supports fp16/bf16 only for this MVP.");
  }

  return out;
}

