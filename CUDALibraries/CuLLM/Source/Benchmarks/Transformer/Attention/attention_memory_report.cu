//------------------------------------------------------------------------------
/// Memory report for the attention ladder: the two memory stories the timing
/// benchmarks don't show.
///
/// 1. HBM working set, MEASURED (cudaMemGetInfo around real cudaMallocs),
///    for the flash path (Q, K, V, O only — identical across every ladder
///    rung by design) vs. the standard-attention baseline (adds TWO
///    B·H·N² workspaces: scores and weights). At N = 4096, B·H = 96, fp32,
///    each N² workspace is ~6.4 GB — the standard path's allocation FAILS
///    live on a 12 GB card. That failure is the demonstration: the memory
///    wall measured, not asserted from arithmetic.
///
/// 2. On-chip resources per kernel (cudaFuncGetAttributes +
///    cudaOccupancyMaxActiveBlocksPerMultiprocessor): static shared memory
///    per block, registers per thread, and achievable blocks/SM. This is
///    where the ladder rungs genuinely differ — HBM footprint does not
///    change from scalar to WMMA to CuTe, but the shared-memory and
///    register budgets (and therefore occupancy) do. This is the
///    memory-hierarchy story a GPU engineer actually asks about.
///
/// Output is plain aligned text, screenshot-ready, with the device name in
/// the header (provenance in frame, matching the repo's benchmark
/// conventions).
//------------------------------------------------------------------------------

#include "Transformer/Attention/attention_scores.h"
#include "Transformer/Attention/attention_weighted_values.h"
#if defined(CULLM_HAS_CUTLASS)
#include "Transformer/Attention/flash_attention_cute.h"
#endif
#include "Transformer/Attention/flash_attention_forward.h"
#include "Transformer/Attention/flash_attention_tensor_core.h"
#include "Transformer/Attention/flash_attention_warp_cooperative.h"

#include <cstdio>
#include <cuda_fp16.h>
#include <vector>

namespace
{

constexpr int kHD {64};
constexpr int kBatchHeads {96};

double to_gib(const size_t bytes)
{
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

//------------------------------------------------------------------------------
// Allocate a list of buffer sizes, measuring actual device-memory delta via
// cudaMemGetInfo. Frees everything before returning. Reports the first
// allocation failure honestly instead of aborting — at the largest shapes
// the failure IS the result.
//------------------------------------------------------------------------------
void measure_working_set(
  const char* label,
  const std::vector<size_t>& buffer_bytes)
{
  size_t free_before {};
  size_t total {};
  cudaMemGetInfo(&free_before, &total);

  std::vector<void*> pointers;
  size_t requested {0};
  bool failed {false};
  for (const size_t bytes : buffer_bytes)
  {
    requested += bytes;
    void* pointer {nullptr};
    const cudaError_t status {cudaMalloc(&pointer, bytes)};
    if (status != cudaSuccess)
    {
      // Clear the sticky error so later allocations are unaffected.
      cudaGetLastError();
      std::printf(
        "  %-34s requested %7.3f GiB -> cudaMalloc FAILED (%s) after %.3f "
        "GiB of it — this failure is the memory wall, measured\n",
        label, to_gib(requested), cudaGetErrorString(status),
        to_gib(requested - bytes));
      failed = true;
      break;
    }
    pointers.push_back(pointer);
  }

  if (!failed)
  {
    size_t free_after {};
    cudaMemGetInfo(&free_after, &total);
    std::printf(
      "  %-34s requested %7.3f GiB -> measured device-memory delta %7.3f "
      "GiB (free: %.3f -> %.3f GiB)\n",
      label, to_gib(requested), to_gib(free_before - free_after),
      to_gib(free_before), to_gib(free_after));
  }

  for (void* pointer : pointers)
  {
    cudaFree(pointer);
  }
}

//------------------------------------------------------------------------------
// Static shared memory, registers, and occupancy for one kernel at its
// production block size.
//------------------------------------------------------------------------------
void report_kernel(
  const char* label,
  const void* kernel,
  const int block_size)
{
  cudaFuncAttributes attributes {};
  if (cudaFuncGetAttributes(&attributes, kernel) != cudaSuccess)
  {
    std::printf("  %-38s (cudaFuncGetAttributes failed)\n", label);
    return;
  }

  int blocks_per_sm {0};
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm, kernel, block_size, 0);

  std::printf(
    "  %-38s %8zu B smem/block  %4d regs/thread  block=%3d  ->  %d "
    "blocks/SM (%4d threads/SM)\n",
    label,
    attributes.sharedSizeBytes,
    attributes.numRegs,
    block_size,
    blocks_per_sm,
    blocks_per_sm * block_size);
}

} // namespace

int main()
{
  cudaDeviceProp properties {};
  cudaGetDeviceProperties(&properties, 0);
  size_t free_bytes {};
  size_t total_bytes {};
  cudaMemGetInfo(&free_bytes, &total_bytes);
  std::printf(
    "Device: %s (sm_%d%d) | VRAM %.2f GiB total, %.2f GiB free | "
    "B*H = %d, d = %d, fp32 element = 4 B\n\n",
    properties.name, properties.major, properties.minor,
    to_gib(total_bytes), to_gib(free_bytes), kBatchHeads, kHD);

  //----------------------------------------------------------------------------
  std::printf(
    "== 1. HBM working set, measured ==\n"
    "Flash kernels (every ladder rung: scalar, WMMA, CuTe) allocate only\n"
    "Q, K, V, O. The standard baseline additionally allocates TWO B*H*N^2\n"
    "workspaces (scores + softmax weights).\n\n");

  for (const int n : {1024, 2048, 4096})
  {
    const size_t qkvo {
      4ull * kBatchHeads * n * kHD * sizeof(float)};
    const size_t n_squared {
      static_cast<size_t>(kBatchHeads) * n * static_cast<size_t>(n) *
        sizeof(float)};

    std::printf("N = %d:\n", n);
    measure_working_set(
      "flash (any rung): Q,K,V,O", {qkvo});
    measure_working_set(
      "standard: Q,K,V,O + S + P", {qkvo, n_squared, n_squared});
    std::printf(
      "  (analytic: each N^2 workspace = %.3f GiB; flash has none)\n\n",
      to_gib(n_squared));
  }

  //----------------------------------------------------------------------------
  std::printf(
    "== 2. On-chip memory per kernel (where the ladder rungs actually "
    "differ) ==\n"
    "HBM footprint is identical across rungs; shared memory, registers,\n"
    "and occupancy are not.\n\n");

  using namespace Transformer::Attention;

  report_kernel(
    "flash thread-per-row (fp32)",
    reinterpret_cast<const void*>(
      &flash_attention_forward<float, kHD, 64, 32, false>),
    64);
  report_kernel(
    "flash warp-cooperative (fp32)",
    reinterpret_cast<const void*>(
      &flash_attention_forward_warp_cooperative<float, kHD, 4, false>),
    128);
  report_kernel(
    "flash warp-cooperative (fp16)",
    reinterpret_cast<const void*>(
      &flash_attention_forward_warp_cooperative<__half, kHD, 4, false>),
    128);
  report_kernel(
    "WMMA tensor-core (fp16)",
    reinterpret_cast<const void*>(
      &flash_attention_forward_tensor_core<__half, kHD, 4, false>),
    128);
#if defined(CULLM_HAS_CUTLASS)
  report_kernel(
    "CuTe/CUTLASS (fp16)",
    reinterpret_cast<const void*>(
      &flash_attention_forward_cute<__half, kHD, 4, false>),
    128);
#endif
  report_kernel(
    "standard: attention_scores (fp32)",
    reinterpret_cast<const void*>(
      &attention_scores<float, kHD, false>),
    128);

  std::printf(
    "\nReading the table: more shared memory / registers per block means\n"
    "fewer resident blocks per SM (lower occupancy) — the tensor-core\n"
    "kernels deliberately spend on-chip memory to avoid HBM round trips.\n"
    "The trade is measured above and in WarpAttentionBenchmark's timings.\n");

  return 0;
}
