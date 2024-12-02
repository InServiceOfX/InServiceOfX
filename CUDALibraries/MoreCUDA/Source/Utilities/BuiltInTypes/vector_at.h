namespace Utilities
{
namespace BuiltInTypes
{

template <typename T4, typename T>
__device__ T vector_at(const T4& input_vector, const unsigned int i) = delete;

//------------------------------------------------------------------------------
/// It's very important to use the inline keyword for specializations to
/// linking errors due to multiple definitions.
/// Otherwise, you may get these erros when trying to include this in another
/// project:
/// [ 87%] Building CUDA object UnitTests/CMakeFiles/Check.dir/LLM/AttentionForward/softmax_tests.cu.o
/// [ 93%] Linking CUDA device code CMakeFiles/Check.dir/cmake_device_link.o
/// nvlink error   : Multiple definition of '_ZN9Utilities12BuiltInTypes9vector_atI7double4dEET0_RKT_j' in 'CMakeFiles/Check.dir/LLM/AttentionForward/softmax_tests.cu.o', first defined in 'CMakeFiles/Check.dir/Drafts/LLM/AttentionForward/softmax_tests.cu.o' (target: sm_52)
/// nvlink error   : Multiple definition of '_ZN9Utilities12BuiltInTypes9vector_atI6float4fEET0_RKT_j' in 'CMakeFiles/Check.dir/LLM/AttentionForward/softmax_tests.cu.o', first defined in 'CMakeFiles/Check.dir/Drafts/LLM/AttentionForward/softmax_tests.cu.o' (target: sm_52)
/// nvlink fatal   : merge_elf failed (target: sm_52)
/// make[2]: *** [UnitTests/CMakeFiles/Check.dir/build.make:139: UnitTests/CMakeFiles/Check.dir/cmake_device_link.o] Error 1
/// make[1]: *** [CMakeFiles/Makefile2:279: UnitTests/CMakeFiles/Check.dir/all] Error 2
/// make: *** [Makefile:136: all] Error 2
//------------------------------------------------------------------------------

template <>
__device__ inline float vector_at(const float4& input_vector, const unsigned int i)
{
  return reinterpret_cast<const float*>(&input_vector)[i];
}

template <>
__device__ inline double vector_at(const double4& input_vector, const unsigned int i)
{
  return reinterpret_cast<const double*>(&input_vector)[i];
}

} // namespace BuiltInTypes
} // namespace Utilities