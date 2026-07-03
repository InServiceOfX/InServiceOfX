//------------------------------------------------------------------------------
/// Multi-head warp-cooperative FlashAttention benchmark for the JAX/XLA
/// comparison (see CUDALibraries/CuLLM/Python/benchmark_report.py).
///
/// The AttentionIOBenchmark times a single (batch, head) slice, which
/// understates every implementation's throughput: real transformer layers
/// run B·H independent attention problems at once, and both this kernel
/// (gridDim.y) and XLA/cuDNN exploit that parallelism. This benchmark
/// sweeps sequence length at a fixed realistic slice count
/// (B·H = 96 ≈ batch 8 × 12 heads, d = 64 — GPT-2-medium-like) in float32
/// and __half, causal and not, and prints one machine-parseable line per
/// configuration:
///
///   CSV dtype,n,batch_heads,causal,mean_ms
///
/// Timing follows the repo convention: cudaEvent around kRepeats launches
/// after kWarmups warmup launches.
//------------------------------------------------------------------------------

#include "DataStructures/Array.h"
#if defined(CULLM_HAS_CUTLASS)
#include "Transformer/Attention/flash_attention_cute.h"
#endif
#include "Transformer/Attention/flash_attention_tensor_core.h"
#include "Transformer/Attention/flash_attention_warp_cooperative.h"

#include <cstdio>
#include <cuda_fp16.h>
#include <functional>
#include <vector>

using DataStructures::Array;
using std::vector;
using Transformer::Attention::flash_attention_tensor_core;
using Transformer::Attention::flash_attention_warp_cooperative;

namespace
{

constexpr int kHD {64};
constexpr int kWarpsPerBlock {4};
constexpr int kBatchHeads {96};
constexpr int kWarmups {3};
constexpr int kRepeats {20};

float time_launches(const std::function<void()>& launch)
{
  for (int w {0}; w < kWarmups; ++w)
  {
    launch();
  }
  cudaDeviceSynchronize();

  cudaEvent_t start {};
  cudaEvent_t stop {};
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int r {0}; r < kRepeats; ++r)
  {
    launch();
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms {0.0f};
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsed_ms / static_cast<float>(kRepeats);
}

vector<float> make_inputs(const int count, const int seed)
{
  vector<float> result(count);
  for (int i {0}; i < count; ++i)
  {
    result[i] =
      static_cast<float>((i * seed + 7 * (i % 13)) % 211 - 105) / 105.0f;
  }
  return result;
}

template <typename T>
vector<T> convert(const vector<float>& values)
{
  vector<T> result(values.size());
  for (size_t i {0}; i < values.size(); ++i)
  {
    result[i] = static_cast<T>(values[i]);
  }
  return result;
}

template <typename T>
void benchmark_type(const char* dtype_name, const int sequence_length)
{
  const long long elements {
    static_cast<long long>(kBatchHeads) * sequence_length * kHD};

  Array<T> d_queries(elements);
  Array<T> d_keys(elements);
  Array<T> d_values(elements);
  Array<T> d_output(elements);

  {
    const vector<float> host_values {
      make_inputs(static_cast<int>(elements), 3)};
    vector<T> converted {convert<T>(host_values)};
    d_queries.copy_host_input_to_device(converted);
    d_keys.copy_host_input_to_device(converted);
    d_values.copy_host_input_to_device(converted);
  }

  const float plain_ms {time_launches([&]()
  {
    flash_attention_warp_cooperative<T, kHD, kWarpsPerBlock, false>(
      d_output.elements_,
      nullptr,
      d_queries.elements_,
      d_keys.elements_,
      d_values.elements_,
      sequence_length,
      kBatchHeads);
  })};
  const float causal_ms {time_launches([&]()
  {
    flash_attention_warp_cooperative<T, kHD, kWarpsPerBlock, true>(
      d_output.elements_,
      nullptr,
      d_queries.elements_,
      d_keys.elements_,
      d_values.elements_,
      sequence_length,
      kBatchHeads);
  })};

  std::printf(
    "CSV %s,%d,%d,0,%.4f\n", dtype_name, sequence_length, kBatchHeads,
    plain_ms);
  std::printf(
    "CSV %s,%d,%d,1,%.4f\n", dtype_name, sequence_length, kBatchHeads,
    causal_ms);
}

void benchmark_tensor_core(const int sequence_length)
{
  const long long elements {
    static_cast<long long>(kBatchHeads) * sequence_length * kHD};

  Array<__half> d_queries(elements);
  Array<__half> d_keys(elements);
  Array<__half> d_values(elements);
  Array<__half> d_output(elements);

  {
    const vector<float> host_values {
      make_inputs(static_cast<int>(elements), 3)};
    vector<__half> converted {convert<__half>(host_values)};
    d_queries.copy_host_input_to_device(converted);
    d_keys.copy_host_input_to_device(converted);
    d_values.copy_host_input_to_device(converted);
  }

  const float plain_ms {time_launches([&]()
  {
    flash_attention_tensor_core<__half, kHD, kWarpsPerBlock, false>(
      d_output.elements_,
      nullptr,
      d_queries.elements_,
      d_keys.elements_,
      d_values.elements_,
      sequence_length,
      kBatchHeads);
  })};
  const float causal_ms {time_launches([&]()
  {
    flash_attention_tensor_core<__half, kHD, kWarpsPerBlock, true>(
      d_output.elements_,
      nullptr,
      d_queries.elements_,
      d_keys.elements_,
      d_values.elements_,
      sequence_length,
      kBatchHeads);
  })};

  std::printf(
    "CSV wmma16,%d,%d,0,%.4f\n", sequence_length, kBatchHeads, plain_ms);
  std::printf(
    "CSV wmma16,%d,%d,1,%.4f\n", sequence_length, kBatchHeads, causal_ms);
}

#if defined(CULLM_HAS_CUTLASS)
void benchmark_cute(const int sequence_length)
{
  const long long elements {
    static_cast<long long>(kBatchHeads) * sequence_length * kHD};

  Array<__half> d_queries(elements);
  Array<__half> d_keys(elements);
  Array<__half> d_values(elements);
  Array<__half> d_output(elements);

  {
    const vector<float> host_values {
      make_inputs(static_cast<int>(elements), 3)};
    vector<__half> converted {convert<__half>(host_values)};
    d_queries.copy_host_input_to_device(converted);
    d_keys.copy_host_input_to_device(converted);
    d_values.copy_host_input_to_device(converted);
  }

  const float plain_ms {time_launches([&]()
  {
    Transformer::Attention::flash_attention_cute<
      __half, kHD, kWarpsPerBlock, false>(
        d_output.elements_,
        nullptr,
        d_queries.elements_,
        d_keys.elements_,
        d_values.elements_,
        sequence_length,
        kBatchHeads);
  })};
  const float causal_ms {time_launches([&]()
  {
    Transformer::Attention::flash_attention_cute<
      __half, kHD, kWarpsPerBlock, true>(
        d_output.elements_,
        nullptr,
        d_queries.elements_,
        d_keys.elements_,
        d_values.elements_,
        sequence_length,
        kBatchHeads);
  })};

  std::printf(
    "CSV cute,%d,%d,0,%.4f\n", sequence_length, kBatchHeads, plain_ms);
  std::printf(
    "CSV cute,%d,%d,1,%.4f\n", sequence_length, kBatchHeads, causal_ms);
}
#endif // CULLM_HAS_CUTLASS

} // namespace

int main()
{
  cudaDeviceProp properties {};
  cudaGetDeviceProperties(&properties, 0);
  std::printf(
    "Device: %s | d = %d, warps/block = %d, batch*heads = %d | "
    "%d repeats\n",
    properties.name, kHD, kWarpsPerBlock, kBatchHeads, kRepeats);

  for (const int sequence_length : {256, 512, 1024, 2048, 4096})
  {
    benchmark_type<float>("float32", sequence_length);
    benchmark_type<__half>("float16", sequence_length);
    benchmark_tensor_core(sequence_length);
#if defined(CULLM_HAS_CUTLASS)
    benchmark_cute(sequence_length);
#endif
  }
  return 0;
}
