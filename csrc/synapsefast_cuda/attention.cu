#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <unordered_map>
#include <mutex>

// FlashAttention-like attention (streaming online softmax).
//
// This CUDA kernel path is a correctness-oriented "FlashAttention-style" online
// softmax. It also includes a faster warp-shuffle reduction specialization for
// D=32 and D=64, which removes the heavy shared-memory + repeated __syncthreads
// reduction used in the earlier prototype.

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

template <typename scalar_t, int HEAD_DIM>
__global__ void attention_fwd_kernel_warp_reduce(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    int T,
    bool causal,
    float scale) {
  // One block per query position for one (batch, head).
  int idx = blockIdx.x; // [B*H*T]
  int qpos = idx % T;
  int bh = idx / T; // [B*H]

  int tid = threadIdx.x; // [0..HEAD_DIM)

  static_assert(HEAD_DIM == 32 || HEAD_DIM == 64, "warp_reduce kernel supports HEAD_DIM=32/64");

  __shared__ float m_s;
  __shared__ float l_s;
  __shared__ float scale_prev_s;
  __shared__ float exp_s;
  __shared__ float score_s;
  __shared__ float sum_warp_s[2]; // used for HEAD_DIM=64 only

  // Load q element once into register.
  float q_val = load_as_float<scalar_t>(q[(bh * T + qpos) * HEAD_DIM + tid]);

  float o_acc = 0.0f;

  if (tid == 0) {
    m_s = -INFINITY;
    l_s = 0.0f;
  }
  __syncthreads();

  for (int t = 0; t < T; ++t) {
    float partial = q_val * load_as_float<scalar_t>(k[(bh * T + t) * HEAD_DIM + tid]);

    float score_sum = 0.0f;

    if (HEAD_DIM == 32) {
      // Single warp reduction.
      for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
      }
      score_sum = partial;
      if (tid == 0) score_s = score_sum;
    } else {
      // HEAD_DIM == 64: two warps.
      int lane = tid % 32;
      int warp = tid / 32;
      for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
      }
      if (lane == 0) sum_warp_s[warp] = partial;
      __syncthreads();
      if (tid == 0) score_s = sum_warp_s[0] + sum_warp_s[1];
    }

    if (tid == 0) {
      bool masked = causal && (t > qpos);
      if (masked) {
        scale_prev_s = 1.0f;
        exp_s = 0.0f;
      } else {
        float score = score_s * scale;
        float m_old = m_s;
        float m_new = fmaxf(m_old, score);

        float scale_prev = 1.0f;
        if (m_new != m_old) scale_prev = expf(m_old - m_new);

        m_s = m_new;
        float exp_term = expf(score - m_new);
        l_s = l_s * scale_prev + exp_term;

        scale_prev_s = scale_prev;
        exp_s = exp_term;
      }
    }

    __syncthreads();

    // Update output accumulator for this d.
    if (scale_prev_s != 1.0f) o_acc *= scale_prev_s;
    if (exp_s != 0.0f) {
      o_acc += exp_s * load_as_float<scalar_t>(v[(bh * T + t) * HEAD_DIM + tid]);
    }

    __syncthreads();
  }

  if (tid == 0) {
    float inv_l = 1.0f / (l_s + 1e-6f);
    score_s = inv_l; // reuse score_s
  }
  __syncthreads();

  float y = o_acc * score_s;
  out[(bh * T + qpos) * HEAD_DIM + tid] = store_from_float<scalar_t>(y);
}

// Q-tiling variant for D=64: each block computes 2 query positions (BLOCK_M=2)
// while caching K/V for that tile in shared memory.
template <typename scalar_t>
__global__ void attention_fwd_kernel_q2_warp_reduce_d64(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    int T,
    bool causal,
    float scale) {
  static_assert(true, "D=64 specialization");
  constexpr int HEAD_DIM = 64;
  constexpr int BLOCK_M = 2;

  int bh = blockIdx.x; // [B*H]
  int q_base = blockIdx.y * BLOCK_M; // query tile base

  int tid = threadIdx.x; // [0..128)
  int row = tid / HEAD_DIM; // 0..1
  int d = tid % HEAD_DIM; // 0..63

  int qpos = q_base + row;
  bool q_valid = qpos < T;

  __shared__ float k_shared[HEAD_DIM];
  __shared__ float v_shared[HEAD_DIM];

  __shared__ float m_s[BLOCK_M];
  __shared__ float l_s[BLOCK_M];
  __shared__ float scale_prev_s[BLOCK_M];
  __shared__ float exp_s[BLOCK_M];
  __shared__ float sum_warp_s[BLOCK_M][2]; // [row][warp_in_row], warp_in_row in {0,1}

  float q_val = 0.0f;
  if (q_valid) {
    q_val = load_as_float<scalar_t>(q[(bh * T + qpos) * HEAD_DIM + d]);
  }

  float o_acc = 0.0f;

  if (d == 0) {
    m_s[row] = -INFINITY;
    l_s[row] = 0.0f;
  }
  __syncthreads();

  for (int t = 0; t < T; ++t) {
    // Cache K/V for this t once per block (row==0 loads).
    if (row == 0) {
      k_shared[d] = load_as_float<scalar_t>(k[(bh * T + t) * HEAD_DIM + d]);
      v_shared[d] = load_as_float<scalar_t>(v[(bh * T + t) * HEAD_DIM + d]);
    }
    __syncthreads();

    // Dot reduction for this (row, qpos).
    float partial = (q_valid ? q_val : 0.0f) * k_shared[d];

    int lane = d % 32;      // 0..31
    int warp_in_row = d / 32; // 0 or 1
    for (int offset = 16; offset > 0; offset >>= 1) {
      partial += __shfl_down_sync(0xffffffff, partial, offset);
    }

    if (lane == 0) {
      sum_warp_s[row][warp_in_row] = partial;
    }
    __syncthreads();

    if (d == 0) {
      bool masked = causal && q_valid && (t > qpos);
      if (masked) {
        scale_prev_s[row] = 1.0f;
        exp_s[row] = 0.0f;
      } else if (!q_valid) {
        scale_prev_s[row] = 1.0f;
        exp_s[row] = 0.0f;
      } else {
        float score = (sum_warp_s[row][0] + sum_warp_s[row][1]) * scale;
        float m_old = m_s[row];
        float m_new = fmaxf(m_old, score);

        float scale_prev = 1.0f;
        if (m_new != m_old) scale_prev = expf(m_old - m_new);

        m_s[row] = m_new;
        // l_s = l_old * scale_prev + exp(score - m_new)
        float exp_term = expf(score - m_new);
        l_s[row] = l_s[row] * scale_prev + exp_term;

        scale_prev_s[row] = scale_prev;
        exp_s[row] = exp_term;
      }
    }
    __syncthreads();

    // Update output accumulator.
    o_acc *= scale_prev_s[row];
    if (exp_s[row] != 0.0f) {
      o_acc += exp_s[row] * v_shared[d];
    }

    __syncthreads();
  }

  // Final normalize.
  if (d == 0) {
    // Reuse exp_s for inv_l to avoid another shared array.
    exp_s[row] = 1.0f / (l_s[row] + 1e-6f);
  }
  __syncthreads();

  if (q_valid) {
    out[(bh * T + qpos) * HEAD_DIM + d] = store_from_float<scalar_t>(o_acc * exp_s[row]);
  }
}

static std::mutex g_mask_mutex;

static std::unordered_map<uint64_t, torch::Tensor> g_causal_mask_cache;

static torch::Tensor get_causal_mask_cached(int64_t T, torch::Device device, at::ScalarType dtype, torch::TensorOptions opts) {
  // Key by (device_index, dtype, T). device_index = -1 for CPU but we only use CUDA here.
  int dev_i = device.has_index() ? device.index() : -1;
  uint64_t dev = static_cast<uint64_t>(dev_i);
  uint64_t dt = static_cast<uint64_t>(dtype == at::kHalf ? 0 : 1);
  uint64_t key = (dev << 32) ^ (static_cast<uint64_t>(T) << 1) ^ dt;

  std::lock_guard<std::mutex> guard(g_mask_mutex);
  auto it = g_causal_mask_cache.find(key);
  if (it != g_causal_mask_cache.end()) {
    return it->second;
  }

  auto mask = at::ones({T, T}, opts.dtype(at::kBool).device(device)).triu(1);
  g_causal_mask_cache.emplace(key, mask);
  return mask;
}

// Generic fallback kernel (older prototype).
template <typename scalar_t>
__global__ void attention_fwd_kernel_naive(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    int T,
    int D,
    bool causal,
    float scale);

template <typename scalar_t>
__global__ void attention_fwd_kernel_naive(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    int T,
    int D,
    bool causal,
    float scale) {
  // One block per query position for one (batch, head).
  int idx = blockIdx.x; // [B*H*T]
  int qpos = idx % T;
  int bh = idx / T; // [B*H]

  int tid = threadIdx.x; // [0..blockDim)

  __shared__ float sdata[256];
  __shared__ float m_s;
  __shared__ float l_s;
  __shared__ float scale_s;
  __shared__ float exp_s;
  __shared__ float inv_l_s;

  float q_val = (tid < D) ? load_as_float<scalar_t>(q[(bh * T + qpos) * D + tid]) : 0.0f;
  float o_acc = 0.0f;

  if (tid == 0) {
    m_s = -INFINITY;
    l_s = 0.0f;
  }
  __syncthreads();

  for (int t = 0; t < T; ++t) {
    float partial = 0.0f;
    if (tid < D) {
      float k_val = load_as_float<scalar_t>(k[(bh * T + t) * D + tid]);
      partial = q_val * k_val;
    }
    sdata[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) sdata[tid] += sdata[tid + stride];
      __syncthreads();
    }

    if (tid == 0) {
      bool masked = causal && (t > qpos);
      if (masked) {
        scale_s = 1.0f;
        exp_s = 0.0f;
      } else {
        float score = sdata[0] * scale;
        float m_old = m_s;
        float m_new = fmaxf(m_old, score);

        float scale_prev = 1.0f;
        if (m_new != m_old) scale_prev = expf(m_old - m_new);

        m_s = m_new;
        l_s = l_s * scale_prev + expf(score - m_new);

        scale_s = scale_prev;
        exp_s = expf(score - m_new);
      }
    }
    __syncthreads();

    if (tid < D) {
      if (scale_s != 1.0f) o_acc *= scale_s;
      if (exp_s != 0.0f) {
        float v_val = load_as_float<scalar_t>(v[(bh * T + t) * D + tid]);
        o_acc += exp_s * v_val;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    inv_l_s = 1.0f / (l_s + 1e-6f);
  }
  __syncthreads();

  if (tid < D) {
    out[(bh * T + qpos) * D + tid] = store_from_float<scalar_t>(o_acc * inv_l_s);
  }
}

torch::Tensor attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda() && v.is_cuda(), "k and v must be CUDA tensors");
  TORCH_CHECK(q.dtype() == k.dtype() && q.dtype() == v.dtype(), "q/k/v must have the same dtype");
  TORCH_CHECK(q.dim() == 4, "q must be [B,H,T,D]");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");

  auto B = (int)q.size(0);
  auto H = (int)q.size(1);
  auto T = (int)q.size(2);
  auto D = (int)q.size(3);

  TORCH_CHECK(D <= 128, "MVP attention CUDA kernel supports head_dim <= 128.");

  auto out = torch::empty_like(q);
  int bh = B * H;

  float scale = 1.0f / sqrtf((float)D);
  const int threads = 128;

  // Ensure k/v are contiguous too.
  q = q.contiguous();
  k = k.contiguous();
  v = v.contiguous();

  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  // Fast path for performance experiments:
  // Use tensor-core-friendly batched matmul + softmax + matmul.
  // This is not FlashAttention-style memory efficient, but it gives a
  // much faster baseline than the streaming prototype kernels above.
  if (D == 64 && (q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16)) {
    // scores: [B,H,T,T]
    auto scores = at::matmul(q, k.transpose(-2, -1)) * scale;
    if (causal) {
      auto mask = get_causal_mask_cached(
          T,
          q.device(),
          q.scalar_type(),
          scores.options());
      scores = scores.masked_fill(mask, -INFINITY);
    }
    auto probs = at::softmax(scores, -1);
    auto y = at::matmul(probs, v); // [B,H,T,D]
    return y;
  }

  if (q.scalar_type() == at::kHalf) {
    const int hd = (int)D;
    if (hd == 32) {
      int blocks = bh * T;
      attention_fwd_kernel_warp_reduce<__half, 32><<<blocks, 32, 0, stream>>>(
          reinterpret_cast<__half*>(q.data_ptr<at::Half>()),
          reinterpret_cast<__half*>(k.data_ptr<at::Half>()),
          reinterpret_cast<__half*>(v.data_ptr<at::Half>()),
          reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
          T,
          causal,
          scale);
    } else if (hd == 64) {
      int blocks = bh * T;
      attention_fwd_kernel_warp_reduce<__half, 64><<<blocks, 64, 0, stream>>>(
          reinterpret_cast<__half*>(q.data_ptr<at::Half>()),
          reinterpret_cast<__half*>(k.data_ptr<at::Half>()),
          reinterpret_cast<__half*>(v.data_ptr<at::Half>()),
          reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
          T,
          causal,
          scale);
    } else {
      int blocks = bh * T;
      attention_fwd_kernel_naive<__half><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<__half*>(q.data_ptr<at::Half>()),
          reinterpret_cast<__half*>(k.data_ptr<at::Half>()),
          reinterpret_cast<__half*>(v.data_ptr<at::Half>()),
          reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
          T,
          D,
          causal,
          scale);
    }
  } else if (q.scalar_type() == at::kBFloat16) {
    const int hd = (int)D;
    if (hd == 32) {
      int blocks = bh * T;
      attention_fwd_kernel_warp_reduce<__nv_bfloat16, 32><<<blocks, 32, 0, stream>>>(
          reinterpret_cast<__nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
          T,
          causal,
          scale);
    } else if (hd == 64) {
      int blocks = bh * T;
      attention_fwd_kernel_warp_reduce<__nv_bfloat16, 64><<<blocks, 64, 0, stream>>>(
          reinterpret_cast<__nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
          T,
          causal,
          scale);
    } else {
      int blocks = bh * T;
      attention_fwd_kernel_naive<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<__nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
          T,
          D,
          causal,
          scale);
    }
  } else {
    TORCH_CHECK(false, "attention_forward only supports fp16/bf16 for this MVP CUDA kernel");
  }

  return out;
}

