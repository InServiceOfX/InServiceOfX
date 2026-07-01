#ifndef UTILITIES_MEMORY_STREAMING_STORE_H
#define UTILITIES_MEMORY_STREAMING_STORE_H

#include <cuda_fp16.h>

// Declarations: /usr/local/cuda/include/sm_32_intrinsics.h  lines 378–412
// Implementations: /usr/local/cuda/include/sm_32_intrinsics.hpp lines 456–499
// __half overloads: /usr/local/cuda/include/cuda_fp16.hpp lines 1982–1984
// PTX ISA cache operators:
//   https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-operators

namespace Utilities
{
namespace Memory
{

//------------------------------------------------------------------------------
/// Writes value to the global-memory address ptr using the PTX
/// st.global.cs (store caching streaming) cache operator.
///
/// Cache semantics (from PTX ISA, st instruction cache operators):
///   "The st.cs store cached-streaming operation allocates cache lines with
///    evict-first policy in L2 (and L1 if local) to limit cache pollution by
///    streaming output data; global streaming data bypasses the L1."
///
/// Cache operator table for st (store) instructions:
///   Modifier  Name              L1            L2
///   .wb       write-back        write-back    write-back    (default __stwb)
///   .cg       cache global      bypass        write-back    (__stcg)
///   .cs       cache streaming   bypass        evict-first   (__stcs)  ← this
///   .wt       write-through     bypass        write-through (__stwt)
///
/// Use streaming_store when writing output a kernel will not read back (e.g.,
/// the normalization pass in softmax_warp_fold_reduce). Without this hint, a
/// normal store keeps output in L1 and L2 as if it will be read again soon —
/// wasting cache space that other warps still need for their input reads. The
/// evict-first hint prevents the write from displacing L2 lines still needed
/// by other warps, and the L1 bypass avoids polluting the L1 entirely.
///
/// Requires compute capability >= 3.2.
/// From sm_32_intrinsics.h line 64:
///   #if defined(_NVHPC_CUDA) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 320
//------------------------------------------------------------------------------
template <typename T>
__device__ void streaming_store(T* ptr, const T value) = delete;

//------------------------------------------------------------------------------
/// float → PTX: st.global.cs.f32
/// From sm_32_intrinsics.hpp line 495:
///   asm("st.global.cs.f32 [%0], %1;" :: __LDG_PTR(ptr), "f"(value) : "memory");
/// Declared in sm_32_intrinsics.h line 408:
///   __SM_32_INTRINSICS_DECL__ void __stcs(float *ptr, float value)
//------------------------------------------------------------------------------
template <>
__device__ inline void streaming_store<float>(float* ptr, const float value)
{
  __stcs(ptr, value);
}

//------------------------------------------------------------------------------
/// double → PTX: st.global.cs.f64
/// From sm_32_intrinsics.hpp line 496:
///   asm("st.global.cs.f64 [%0], %1;" :: __LDG_PTR(ptr), "d"(value) : "memory");
/// Declared in sm_32_intrinsics.h line 409:
///   __SM_32_INTRINSICS_DECL__ void __stcs(double *ptr, double value)
//------------------------------------------------------------------------------
template <>
__device__ inline void streaming_store<double>(double* ptr, const double value)
{
  __stcs(ptr, value);
}

//------------------------------------------------------------------------------
/// __half → PTX: st.global.cs.b16
/// From cuda_fp16.hpp line 1982–1984:
///   __CUDA_FP16_DECL__ void __stcs(__half *const ptr, const __half value)
///   {
///     asm("st.global.cs.b16 [%0], %1;" :: __LDG_PTR(ptr), "h"(__HALF_TO_CUS(value)) : "memory");
///   }
//------------------------------------------------------------------------------
template <>
__device__ inline void streaming_store<__half>(__half* ptr, const __half value)
{
  __stcs(ptr, value);
}

} // namespace Memory
} // namespace Utilities

#endif // UTILITIES_MEMORY_STREAMING_STORE_H
