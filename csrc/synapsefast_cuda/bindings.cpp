#include <torch/extension.h>

torch::Tensor attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal);
torch::Tensor kv_cache_prefill_forward(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_new, torch::Tensor v_new, int start_pos);
torch::Tensor kv_cache_decode_forward(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, int pos, bool causal);
torch::Tensor rms_norm_forward(torch::Tensor x, torch::Tensor weight, double eps);
torch::Tensor layer_norm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps);
torch::Tensor gelu_forward(torch::Tensor x);
torch::Tensor bias_add_forward(torch::Tensor x, torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attention_forward", &attention_forward, "attention_forward (CUDA)");
  m.def("kv_cache_prefill_forward", &kv_cache_prefill_forward, "kv_cache_prefill_forward (CUDA)");
  m.def("kv_cache_decode_forward", &kv_cache_decode_forward, "kv_cache_decode_forward (CUDA)");
  m.def("rms_norm_forward", &rms_norm_forward, "rms_norm_forward (CUDA)");
  m.def("layer_norm_forward", &layer_norm_forward, "layer_norm_forward (CUDA)");
  m.def("gelu_forward", &gelu_forward, "gelu_forward (CUDA)");
  m.def("bias_add_forward", &bias_add_forward, "bias_add_forward (CUDA)");
}

