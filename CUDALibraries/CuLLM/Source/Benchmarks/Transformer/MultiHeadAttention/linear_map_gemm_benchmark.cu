//------------------------------------------------------------------------------
/// Linear-map GEMM benchmark: hand-written tiled shared-memory GEMM vs.
/// cuBLASLt, at the shapes the multi-head attention linear maps actually
/// run (see qkv_linear_maps.h and output_linear_map.h):
///
///   QKV projection:    (B·T, d_model) · (d_model, 3·d_model)
///   output projection: (B·T, d_model) · (d_model, d_model)
///
/// This answers the question that originally motivated MultiHeadAttention/:
/// "should we use cuBLASLt or write our own GEMM?" The tiled kernel is the
/// textbook baseline (one shared-memory staging pass per k-tile, no register
/// blocking, no tensor cores); cuBLASLt picks tensor-core kernels through
/// its heuristic. The expected outcome is a wide cuBLASLt win at realistic
/// d_model — the point of measuring is to know the factor.
///
/// Correctness is cross-checked (max |Δ| between the two outputs) before
/// timing, so a fast-but-wrong configuration cannot masquerade as a win.
///
/// Timing harness follows attention_io_benchmark.cu: warmup launches, then
/// cudaEvent-timed repeats, reporting mean per-launch time and GFLOP/s
/// (2·m·k·n FLOPs per GEMM).
//------------------------------------------------------------------------------

#include "cuBLASWrappers/LibraryContextHandle.h"
#include "cuBLASWrappers/MatrixMultiplication/LtMatrixMultiplication.h"
#include "cuBLASWrappers/MatrixMultiplication/Setup.h"
#include "DataStructures/Array.h"
#include "StreamManagement/Stream.h"
#include "Transformer/MultiHeadAttention/tiled_gemm.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

using cuBLASWrappers::LibraryContextHandle;
using DataStructures::Array;
using StreamManagement::Stream;
using std::vector;
using Transformer::MultiHeadAttention::tiled_gemm_launch;

namespace
{

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

//------------------------------------------------------------------------------
// One row of the comparison: row-major Out(m,n) = X(m,k)·W(k,n) via both
// implementations, correctness delta, then timings.
//------------------------------------------------------------------------------
void benchmark_shape(
  LibraryContextHandle& handle,
  Stream& stream,
  const char* label,
  const int m,
  const int k,
  const int n)
{
  const vector<float> input {make_inputs(m * k, 3)};
  const vector<float> weights {make_inputs(k * n, 5)};

  Array<float> d_input(m * k);
  Array<float> d_weights(k * n);
  Array<float> d_output_tiled(m * n);
  Array<float> d_output_cublas(m * n);
  d_input.copy_host_input_to_device(input);
  d_weights.copy_host_input_to_device(weights);

  // cuBLASLt setup once per shape (heuristic + workspace reused across
  // repeats, as in the linear-map forward code).
  cuBLASWrappers::MatrixMultiplication::Setup<float> setup(n, m, k);
  if (!setup.setup(handle))
  {
    std::printf("%s: cuBLASLt setup failed\n", label);
    return;
  }
  cuBLASWrappers::MatrixMultiplication::LtMatrixMultiplication<float>
    matmul{};

  const auto launch_tiled {[&]()
  {
    tiled_gemm_launch<float>(
      d_output_tiled.elements_,
      d_input.elements_,
      d_weights.elements_,
      m, k, n);
  }};
  const auto launch_cublas {[&]()
  {
    matmul(
      handle,
      setup.descriptor_,
      setup.layouts_,
      setup.heuristic_,
      stream,
      setup.workspace_,
      d_weights.elements_,
      d_input.elements_,
      nullptr,
      d_output_cublas.elements_);
  }};

  // Correctness gate before timing.
  launch_tiled();
  launch_cublas();
  cudaDeviceSynchronize();
  vector<float> tiled(static_cast<size_t>(m) * n);
  vector<float> cublas(static_cast<size_t>(m) * n);
  d_output_tiled.copy_device_output_to_host(tiled);
  d_output_cublas.copy_device_output_to_host(cublas);
  float max_delta {0.0f};
  for (size_t i {0}; i < tiled.size(); ++i)
  {
    max_delta = std::fmax(max_delta, std::fabs(tiled[i] - cublas[i]));
  }

  const float tiled_ms {time_launches(launch_tiled)};
  const float cublas_ms {time_launches(launch_cublas)};

  const double flops {2.0 * m * k * n};
  const double tiled_gflops {flops / (tiled_ms * 1e6)};
  const double cublas_gflops {flops / (cublas_ms * 1e6)};

  std::printf(
    "%-28s m=%5d k=%5d n=%5d | tiled %8.3f ms (%7.1f GF/s) | "
    "cuBLASLt %8.3f ms (%7.1f GF/s) | speedup %5.1fx | max|dt| %.2e\n",
    label, m, k, n,
    tiled_ms, tiled_gflops,
    cublas_ms, cublas_gflops,
    tiled_ms / cublas_ms,
    max_delta);
}

} // namespace

int main()
{
  LibraryContextHandle handle {};
  Stream stream {};

  std::printf(
    "Linear-map GEMM: hand-written 32x32 tiled shared-memory kernel vs. "
    "cuBLASLt\n(row-major Out = X W, %d timed repeats after %d warmups)\n\n",
    kRepeats, kWarmups);

  // QKV projection shapes: (B·T, d_model)·(d_model, 3·d_model).
  benchmark_shape(handle, stream, "qkv  d_model=256  B*T=2048", 2048, 256, 768);
  benchmark_shape(handle, stream, "qkv  d_model=512  B*T=2048", 2048, 512, 1536);
  benchmark_shape(handle, stream, "qkv  d_model=768  B*T=2048", 2048, 768, 2304);
  benchmark_shape(handle, stream, "qkv  d_model=1024 B*T=2048", 2048, 1024, 3072);

  // Output projection shapes: (B·T, d_model)·(d_model, d_model).
  benchmark_shape(handle, stream, "out  d_model=512  B*T=2048", 2048, 512, 512);
  benchmark_shape(handle, stream, "out  d_model=1024 B*T=2048", 2048, 1024, 1024);

  return 0;
}
