//------------------------------------------------------------------------------
/// IO-complexity benchmark: standard attention vs. FlashAttention.
///
/// Both algorithms perform Θ(n²d) FLOPs — FlashAttention does not reduce
/// arithmetic; it reduces HBM traffic (see the remark on FLOPs in the
/// section on IO Complexity of FlashAttention in FlashAttention.tex):
///
///   standard:  4n² + 4nd elements moved (S and P are materialized in HBM;
///              see the table in the section on IO Complexity of Standard
///              Attention),
///   flash:     (2·T_r + 2)·n·d elements, T_r = ⌈n/B_r⌉ (Q read once per
///              row block; K, V once per (row block, column tile) pair;
///              S and P never touch HBM).
///
/// Since attention at these sizes is memory-bound, the measured speedup
/// should track the modeled traffic ratio ~(4n²)/(2·T_r·n·d) as n grows.
/// The causal column shows the tile-skipping effect: about half the K/V
/// tiles lie above the diagonal and are never loaded.
///
/// Timing harness follows llm.c's dev/cuda style: warmup launches, then
/// cudaEvent-timed repeats, reporting mean per-launch time.
//------------------------------------------------------------------------------

#include "DataStructures/Array.h"
#include "Transformer/Attention/flash_attention_forward.h"
#include "Transformer/Attention/scaled_dot_product_attention.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

using DataStructures::Array;
using std::vector;
using Transformer::Attention::flash_attention;
using Transformer::Attention::scaled_dot_product_attention;

namespace
{

constexpr int kHD {64};
constexpr int kBr {64};
constexpr int kBc {32};
constexpr int kWarmups {3};
constexpr int kRepeats {20};

//------------------------------------------------------------------------------
// Mean milliseconds per launch over kRepeats, after kWarmups warmup launches.
//------------------------------------------------------------------------------
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

} // namespace

int main()
{
  cudaDeviceProp properties {};
  cudaGetDeviceProperties(&properties, 0);
  std::printf(
    "Device: %s | d = %d, B_r = %d, B_c = %d | %d repeats\n\n",
    properties.name,
    kHD,
    kBr,
    kBc,
    kRepeats);

  std::printf(
    "%6s | %12s | %12s | %8s | %10s | %12s | %10s\n",
    "n",
    "standard ms",
    "flash ms",
    "speedup",
    "IO model",
    "causal ms",
    "max |diff|");
  std::printf(
    "-------+--------------+--------------+----------+------------+--------------+-----------\n");

  for (const int n : {256, 512, 1024, 2048, 4096})
  {
    const vector<float> queries {make_inputs(n * kHD, 3)};
    const vector<float> keys {make_inputs(n * kHD, 5)};
    const vector<float> values {make_inputs(n * kHD, 11)};

    Array<float> d_queries(n * kHD);
    Array<float> d_keys(n * kHD);
    Array<float> d_values(n * kHD);
    Array<float> d_scores(n * n);
    Array<float> d_weights(n * n);
    Array<float> d_standard_output(n * kHD);
    Array<float> d_flash_output(n * kHD);
    d_queries.copy_host_input_to_device(queries);
    d_keys.copy_host_input_to_device(keys);
    d_values.copy_host_input_to_device(values);

    const float standard_ms {time_launches(
      [&]()
      {
        scaled_dot_product_attention<float, kHD>(
          d_standard_output.elements_,
          d_scores.elements_,
          d_weights.elements_,
          d_queries.elements_,
          d_keys.elements_,
          d_values.elements_,
          n);
      })};

    const float flash_ms {time_launches(
      [&]()
      {
        flash_attention<float, kHD, kBr, kBc>(
          d_flash_output.elements_,
          d_queries.elements_,
          d_keys.elements_,
          d_values.elements_,
          n);
      })};

    const float causal_ms {time_launches(
      [&]()
      {
        flash_attention<float, kHD, kBr, kBc, true>(
          d_flash_output.elements_,
          d_queries.elements_,
          d_keys.elements_,
          d_values.elements_,
          n);
      })};

    // Re-run non-causal flash so the exactness check below compares
    // standard vs. flash on identical inputs.
    flash_attention<float, kHD, kBr, kBc>(
      d_flash_output.elements_,
      d_queries.elements_,
      d_keys.elements_,
      d_values.elements_,
      n);
    cudaDeviceSynchronize();

    vector<float> standard_output(n * kHD);
    vector<float> flash_output(n * kHD);
    d_standard_output.copy_device_output_to_host(standard_output);
    d_flash_output.copy_device_output_to_host(flash_output);

    float max_difference {0.0f};
    for (int i {0}; i < n * kHD; ++i)
    {
      max_difference = std::fmax(
        max_difference,
        std::fabs(standard_output[i] - flash_output[i]));
    }

    // Modeled HBM element counts (see header comment).
    const double standard_elements {
      4.0 * static_cast<double>(n) * n + 4.0 * n * kHD};
    const int row_blocks {(n + kBr - 1) / kBr};
    const double flash_elements {
      (2.0 * row_blocks + 2.0) * static_cast<double>(n) * kHD};
    const double io_model_ratio {standard_elements / flash_elements};

    std::printf(
      "%6d | %12.4f | %12.4f | %7.2fx | %9.2fx | %12.4f | %10.2e\n",
      n,
      standard_ms,
      flash_ms,
      standard_ms / flash_ms,
      io_model_ratio,
      causal_ms,
      max_difference);
  }

  return 0;
}
