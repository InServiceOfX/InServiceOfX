#ifndef UTILITIES_MEMORY_STREAMING_LOAD_H
#define UTILITIES_MEMORY_STREAMING_LOAD_H

#include <cuda_fp16.h>

// Declarations: /usr/local/cuda/include/sm_32_intrinsics.h  lines 228–232
// Implementations: /usr/local/cuda/include/sm_32_intrinsics.hpp lines 270–271
// __half overloads: /usr/local/cuda/include/cuda_fp16.hpp lines 1932–1936
// PTX ISA cache operators:
//   https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-operators

namespace Utilities
{
namespace Memory
{

//------------------------------------------------------------------------------
/// Loads and returns the value at global-memory address ptr using the PTX
/// ld.global.cs (load caching streaming) cache operator.
///
/// Cache semantics (from PTX ISA, ld instruction cache operators):
///   "The ld.cs load cached-streaming operation allocates cache lines with
///    evict-first policy in L2 and L1 to limit cache pollution by streaming
///    input data."
///
/// Cache operator table for ld (load) instructions:
///   Modifier  Name              L1            L2
///   .ca       cache all         cache         cache         (default)
///   .cg       cache global      bypass        cache         (__ldcg)
///   .cs       cache streaming   evict-first   evict-first   (__ldcs)  ← this
///   .lu       last use          evict-first   invalidate    (__ldlu)
///   .cv       cache volatile    bypass        bypass        (__ldcv)
///
/// Use streaming_load when reading input data exactly once per kernel. The
/// evict-first hint allows the hardware to reclaim the cache lines soon after
/// the load, freeing capacity for data that other threads still need. In
/// softmax kernels this is used for the second pass over x[] (computing exp),
/// after x[] was already read in the first pass (computing max). Without this
/// hint the second pass would re-fill L1/L2 with x[] lines that will not be
/// read again, displacing the intermediate exp[] output that the third pass
/// needs.
///
/// Requires compute capability >= 3.2.
/// From sm_32_intrinsics.h line 64:
///   #if defined(_NVHPC_CUDA) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 320
//------------------------------------------------------------------------------
template <typename T>
__device__ T streaming_load(const T* ptr) = delete;

//------------------------------------------------------------------------------
/// float → PTX: ld.global.cs.f32
/// From sm_32_intrinsics.hpp line 270:
///   asm volatile ("ld.global.cs.f32 %0, [%1];" : "=f"(ret) : __LDG_PTR(ptr));
/// Declared in sm_32_intrinsics.h line 228:
///   __SM_32_INTRINSICS_DECL__ float __ldcs(const float *ptr)
//------------------------------------------------------------------------------
template <>
__device__ inline float streaming_load<float>(const float* ptr)
{
  return __ldcs(ptr);
}

//------------------------------------------------------------------------------
/// double → PTX: ld.global.cs.f64
/// From sm_32_intrinsics.hpp line 271:
///   asm volatile ("ld.global.cs.f64 %0, [%1];" : "=d"(ret) : __LDG_PTR(ptr));
/// Declared in sm_32_intrinsics.h line 229:
///   __SM_32_INTRINSICS_DECL__ double __ldcs(const double *ptr)
//------------------------------------------------------------------------------
template <>
__device__ inline double streaming_load<double>(const double* ptr)
{
  return __ldcs(ptr);
}

//------------------------------------------------------------------------------
/// __half → PTX: ld.global.cs.b16
/// From cuda_fp16.hpp lines 1932–1936:
///   __CUDA_FP16_DECL__ __half __ldcs(const __half *const ptr)
///   {
///     __half ret;
///     asm ("ld.global.cs.b16 %0, [%1];" : "=h"(__HALF_TO_US(ret)) : __LDG_PTR(ptr));
///     return ret;
///   }
//------------------------------------------------------------------------------
template <>
__device__ inline __half streaming_load<__half>(const __half* ptr)
{
  return __ldcs(ptr);
}

} // namespace Memory
} // namespace Utilities

#endif // UTILITIES_MEMORY_STREAMING_LOAD_H
