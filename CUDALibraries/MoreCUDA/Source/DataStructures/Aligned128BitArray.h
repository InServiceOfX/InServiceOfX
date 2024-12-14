#ifndef DATASTRUCTURES_ALIGNED_128_BIT_ARRAY_H
#define DATASTRUCTURES_ALIGNED_128_BIT_ARRAY_H

#include <cstdint>
#include <cstdlib>

namespace DataStructures
{

// https://en.cppreference.com/w/cpp/language/object#Alignment
// Alignment requirement of 16 bytes; number of bytes between successive
// addresses at which objects of this type can be allocated.
template<typename T>
struct alignas(16) Aligned128BitArray
{
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=int4#char-short-int-long-longlong-float-double
  // int4 has alignment 16. int4 is a vector type. They are structures and 4
  // components.
  static constexpr const uint32_t size_ {
    static_cast<uint32_t>(sizeof(int4) / sizeof(T))};

  __device__ explicit Aligned128BitArray(int4 bits)
  {
    static_assert(sizeof(bits) == sizeof(elements_), "Size mismatch");
    memcpy(&elements_, &bits, sizeof(bits));
  }

  __device__ T& operator[](const uint32_t index)
  {
    return elements_[index];
  }

  __device__ const T operator[](const uint32_t index) const
  {
    return elements_[index];
  }

  __device__ int4 get_as_bits() const
  {
    int4 bits {};
    static_assert(sizeof(bits) == sizeof(elements_), "Size mismatch");
    memcpy(&bits, &elements_, sizeof(bits));
    return bits;
  }

  T elements_[size_];

  Aligned128BitArray() = default;
};

template <typename T>
__device__ Aligned128BitArray<T> load_from_address(const T* address)
{
  return Aligned128BitArray<T>(*reinterpret_cast<const int4*>(address));
}

//------------------------------------------------------------------------------
/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=__ldcs#load-functions-using-cache-hints
/// 7.11. Load Functions Using Cache Hints
/// T __ldcs(const T* address)
/// returns data of type T located at address, where T is char, ..., int4, ...
/// float, float2, float4, double, or double2. With cuda_fp16.h header included,
/// T can be __half or __half2. Similarly, with cuda_bf16.h, T can also be
/// __nv_bfloat16 or __nv_bfloat162. The operation is using the corresponding
/// cache operator (see PTX ISA).
///
/// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=cache%2520operator#cache-operators
/// 9.7.10.1. Cache Operators
/// Cache operators on load or store instructions are treated as performance
/// hints only.
/// .cs - Cache streaming, likely to be accessed once.
/// ld.cs load cached streaming operation allocates global lines with evict-
/// first policy in L1 and L2 to limit cache pollution by temporary streaming
/// data that may be accessed once or twice.
//------------------------------------------------------------------------------
template <typename T>
__device__ Aligned128BitArray<T> load_with_cache_streaming_hint(
  const T* address)
{
  return Aligned128BitArray<T>(__ldcs(reinterpret_cast<const int4*>(address)));
}

template <typename T>
__device__ void store_to_address(
  T* address,
  Aligned128BitArray<T> aligned_array)
{
  *reinterpret_cast<int4*>(address) = aligned_array.get_as_bits();
}

} // namespace DataStructures

#endif // DATASTRUCTURES_ALIGNED_128_BIT_ARRAY_H