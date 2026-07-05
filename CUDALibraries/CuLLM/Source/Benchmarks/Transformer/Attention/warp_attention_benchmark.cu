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
/// Timing follows the repo convention: cudaEvent around repeated launches
/// after warmup launches. The CSV lines are kept stable for Python parsers;
/// the readable summary printed after them is for presentations and quick
/// inspection.
//------------------------------------------------------------------------------

#include "DataStructures/Array.h"
#if defined(CULLM_HAS_CUTLASS)
#include "Transformer/Attention/flash_attention_cute.h"
#endif
#include "Transformer/Attention/flash_attention_tensor_core.h"
#include "Transformer/Attention/flash_attention_warp_cooperative.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <functional>
#include <string>
#include <vector>

using DataStructures::Array;
using std::vector;
using Transformer::Attention::flash_attention_tensor_core;
using Transformer::Attention::flash_attention_warp_cooperative;

namespace
{

constexpr int kHD {64};
constexpr int kWarpsPerBlock {4};
constexpr int kDefaultBatchHeads {96};
constexpr int kDefaultWarmups {3};
constexpr int kDefaultRepeats {20};

struct Timings
{
  float non_causal_ms {};
  float causal_ms {};
};

struct BenchmarkRow
{
  int sequence_length {};
  Timings float32 {};
  Timings float16 {};
  Timings wmma16 {};
  Timings cute {};
  bool has_cute {false};
};

struct Options
{
  int batch_heads {kDefaultBatchHeads};
  int warmups {kDefaultWarmups};
  int repeats {kDefaultRepeats};
  bool csv_only {false};
  bool stress {false};
  bool repeats_was_set {false};
};

float time_launches(
  const std::function<void()>& launch,
  const int warmups,
  const int repeats)
{
  for (int w {0}; w < warmups; ++w)
  {
    launch();
  }
  cudaDeviceSynchronize();

  cudaEvent_t start {};
  cudaEvent_t stop {};
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int r {0}; r < repeats; ++r)
  {
    launch();
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms {0.0f};
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsed_ms / static_cast<float>(repeats);
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
Timings benchmark_type(
  const char* dtype_name,
  const int sequence_length,
  const int batch_heads,
  const int warmups,
  const int repeats)
{
  const long long elements {
    static_cast<long long>(batch_heads) * sequence_length * kHD};

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
      batch_heads);
  }, warmups, repeats)};
  const float causal_ms {time_launches([&]()
  {
    flash_attention_warp_cooperative<T, kHD, kWarpsPerBlock, true>(
      d_output.elements_,
      nullptr,
      d_queries.elements_,
      d_keys.elements_,
      d_values.elements_,
      sequence_length,
      batch_heads);
  }, warmups, repeats)};

  std::printf(
    "CSV %s,%d,%d,0,%.4f\n", dtype_name, sequence_length, batch_heads,
    plain_ms);
  std::printf(
    "CSV %s,%d,%d,1,%.4f\n", dtype_name, sequence_length, batch_heads,
    causal_ms);
  return {plain_ms, causal_ms};
}

Timings benchmark_tensor_core(
  const int sequence_length,
  const int batch_heads,
  const int warmups,
  const int repeats)
{
  const long long elements {
    static_cast<long long>(batch_heads) * sequence_length * kHD};

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
      batch_heads);
  }, warmups, repeats)};
  const float causal_ms {time_launches([&]()
  {
    flash_attention_tensor_core<__half, kHD, kWarpsPerBlock, true>(
      d_output.elements_,
      nullptr,
      d_queries.elements_,
      d_keys.elements_,
      d_values.elements_,
      sequence_length,
      batch_heads);
  }, warmups, repeats)};

  std::printf(
    "CSV wmma16,%d,%d,0,%.4f\n", sequence_length, batch_heads, plain_ms);
  std::printf(
    "CSV wmma16,%d,%d,1,%.4f\n", sequence_length, batch_heads, causal_ms);
  return {plain_ms, causal_ms};
}

#if defined(CULLM_HAS_CUTLASS)
Timings benchmark_cute(
  const int sequence_length,
  const int batch_heads,
  const int warmups,
  const int repeats)
{
  const long long elements {
    static_cast<long long>(batch_heads) * sequence_length * kHD};

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
        batch_heads);
  }, warmups, repeats)};
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
        batch_heads);
  }, warmups, repeats)};

  std::printf(
    "CSV cute,%d,%d,0,%.4f\n", sequence_length, batch_heads, plain_ms);
  std::printf(
    "CSV cute,%d,%d,1,%.4f\n", sequence_length, batch_heads, causal_ms);
  return {plain_ms, causal_ms};
}
#endif // CULLM_HAS_CUTLASS

float ratio(const float numerator, const float denominator)
{
  return denominator > 0.0f ? numerator / denominator : 0.0f;
}

void print_help(const char* program)
{
  std::printf(
    "Usage: %s [--batch-heads N] [--repeats N] [--warmups N] "
    "[--stress] [--csv-only]\n\n"
    "Fixed by the compiled kernels: d = 64, warps/block = 4.\n"
    "  --batch-heads N  Runtime B*H slice count (default 96).\n"
    "  --repeats N      Timed launches per row (default 20; stress default 5).\n"
    "  --warmups N      Warmup launches per row (default 3).\n"
    "  --stress         Add N = 8192 to the sweep. Expect minutes on RTX 3060.\n"
    "  --csv-only       Suppress the readable summary; keep parser CSV only.\n",
    program);
}

int parse_positive_int(const char* text, const char* flag)
{
  char* end {};
  const long value {std::strtol(text, &end, 10)};
  if (end == text || *end != '\0' || value <= 0)
  {
    std::fprintf(stderr, "Invalid %s value: %s\n", flag, text);
    std::exit(2);
  }
  return static_cast<int>(value);
}

Options parse_options(const int argc, char** argv)
{
  Options options {};
  for (int i {1}; i < argc; ++i)
  {
    if (std::strcmp(argv[i], "--help") == 0 ||
      std::strcmp(argv[i], "-h") == 0)
    {
      print_help(argv[0]);
      std::exit(0);
    }
    else if (std::strcmp(argv[i], "--csv-only") == 0)
    {
      options.csv_only = true;
    }
    else if (std::strcmp(argv[i], "--stress") == 0)
    {
      options.stress = true;
    }
    else if (std::strcmp(argv[i], "--batch-heads") == 0 && i + 1 < argc)
    {
      options.batch_heads = parse_positive_int(argv[++i], "--batch-heads");
    }
    else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc)
    {
      options.repeats = parse_positive_int(argv[++i], "--repeats");
      options.repeats_was_set = true;
    }
    else if (std::strcmp(argv[i], "--warmups") == 0 && i + 1 < argc)
    {
      options.warmups = parse_positive_int(argv[++i], "--warmups");
    }
    else
    {
      std::fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
      print_help(argv[0]);
      std::exit(2);
    }
  }
  if (options.stress && !options.repeats_was_set)
  {
    options.repeats = 5;
  }
  return options;
}

void print_summary(
  const vector<BenchmarkRow>& rows,
  const Options& options)
{
  std::printf("\nReadable summary\n");
  std::printf("Fixed kernel shape: d = %d per head, warps/block = %d. "
    "Runtime slices: batch*heads = %d.\n", kHD, kWarpsPerBlock,
    options.batch_heads);
  std::printf("Mean ms over %d timed launches after %d warmups.\n\n",
    options.repeats, options.warmups);

  std::printf(
    "Non-causal fp16 engine ladder\n"
    "| N | scalar fp16 | WMMA | CuTe | scalar/CuTe | WMMA/CuTe |\n"
    "|---|---:|---:|---:|---:|---:|\n");
  for (const auto& row : rows)
  {
    if (row.has_cute)
    {
      std::printf(
        "| %d | %.2f | %.2f | %.2f | %.1fx | %.1fx |\n",
        row.sequence_length,
        row.float16.non_causal_ms,
        row.wmma16.non_causal_ms,
        row.cute.non_causal_ms,
        ratio(row.float16.non_causal_ms, row.cute.non_causal_ms),
        ratio(row.wmma16.non_causal_ms, row.cute.non_causal_ms));
    }
  }

  std::printf(
    "\nCausal tile skipping\n"
    "| N | float32 non-causal | float32 causal | speedup | "
    "CuTe non-causal | CuTe causal | speedup |\n"
    "|---|---:|---:|---:|---:|---:|---:|\n");
  for (const auto& row : rows)
  {
    if (row.has_cute)
    {
      std::printf(
        "| %d | %.2f | %.2f | %.2fx | %.2f | %.2f | %.2fx |\n",
        row.sequence_length,
        row.float32.non_causal_ms,
        row.float32.causal_ms,
        ratio(row.float32.non_causal_ms, row.float32.causal_ms),
        row.cute.non_causal_ms,
        row.cute.causal_ms,
        ratio(row.cute.non_causal_ms, row.cute.causal_ms));
    }
  }

  if (!rows.empty())
  {
    const BenchmarkRow& row {rows[rows.size() > 3 ? 3 : rows.size() - 1]};
    if (row.has_cute)
    {
      std::printf(
        "\nHeadline at N=%d: scalar fp16 %.1f ms -> WMMA %.1f ms -> "
        "CuTe %.1f ms (%.1fx faster than scalar, %.1fx faster than WMMA).\n",
        row.sequence_length,
        row.float16.non_causal_ms,
        row.wmma16.non_causal_ms,
        row.cute.non_causal_ms,
        ratio(row.float16.non_causal_ms, row.cute.non_causal_ms),
        ratio(row.wmma16.non_causal_ms, row.cute.non_causal_ms));
    }
  }

  if (options.stress && rows.size() >= 2)
  {
    const BenchmarkRow* n4096 {};
    const BenchmarkRow* n8192 {};
    for (const auto& row : rows)
    {
      if (row.sequence_length == 4096)
      {
        n4096 = &row;
      }
      else if (row.sequence_length == 8192)
      {
        n8192 = &row;
      }
    }
    if (n4096 != nullptr && n8192 != nullptr && n8192->has_cute)
    {
      std::printf(
        "\nStress note: N doubles 4096 -> 8192, but exact dense attention "
        "work is quadratic. CuTe non-causal %.1f -> %.1f ms (%.2fx); "
        "CuTe causal %.1f -> %.1f ms (%.2fx). Causal stays near 2x "
        "faster because future tiles are skipped.\n",
        n4096->cute.non_causal_ms,
        n8192->cute.non_causal_ms,
        ratio(n8192->cute.non_causal_ms, n4096->cute.non_causal_ms),
        n4096->cute.causal_ms,
        n8192->cute.causal_ms,
        ratio(n8192->cute.causal_ms, n4096->cute.causal_ms));
    }
  }
}

} // namespace

int main(int argc, char** argv)
{
  const Options options {parse_options(argc, argv)};
  cudaDeviceProp properties {};
  cudaGetDeviceProperties(&properties, 0);
  std::printf(
    "Device: %s | d = %d, warps/block = %d, batch*heads = %d | "
    "%d repeats\n",
    properties.name, kHD, kWarpsPerBlock, options.batch_heads,
    options.repeats);

  vector<int> sequence_lengths {256, 512, 1024, 2048, 4096};
  if (options.stress)
  {
    sequence_lengths.push_back(8192);
  }

  vector<BenchmarkRow> rows;
  rows.reserve(sequence_lengths.size());
  for (const int sequence_length : sequence_lengths)
  {
    BenchmarkRow row {};
    row.sequence_length = sequence_length;
    row.float32 = benchmark_type<float>(
      "float32",
      sequence_length,
      options.batch_heads,
      options.warmups,
      options.repeats);
    row.float16 = benchmark_type<__half>(
      "float16",
      sequence_length,
      options.batch_heads,
      options.warmups,
      options.repeats);
    row.wmma16 = benchmark_tensor_core(
      sequence_length,
      options.batch_heads,
      options.warmups,
      options.repeats);
#if defined(CULLM_HAS_CUTLASS)
    row.cute = benchmark_cute(
      sequence_length,
      options.batch_heads,
      options.warmups,
      options.repeats);
    row.has_cute = true;
#endif
    rows.push_back(row);
  }
  if (!options.csv_only)
  {
    print_summary(rows, options);
  }
  return 0;
}
