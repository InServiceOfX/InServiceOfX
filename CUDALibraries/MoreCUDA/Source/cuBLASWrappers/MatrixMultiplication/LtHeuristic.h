#ifndef CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_LT_HEURISTIC_RESULT_H
#define CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_LT_HEURISTIC_RESULT_H

#include "cuBLASWrappers/LibraryContextHandle.h"
#include "cuBLASWrappers/MatrixMultiplication/LtDescriptor.h"
#include "cuBLASWrappers/MatrixMultiplication/LtLayouts.h"
#include "cuBLASWrappers/MatrixMultiplication/LtPreference.h"

#include <cublasLt.h>

namespace cuBLASWrappers
{
namespace MatrixMultiplication
{
//------------------------------------------------------------------------------
/// \brief cuBLASLt ("lightweight"?) heuristic.
/// https://docs.nvidia.com/cuda/cublas/index.html?highlight=CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES#heuristics-cache
/// 3.1.2. Heuristics Cache
/// cuBLASLt uses heuristics to pick the most suitable matmul kernel for
/// execution based on problem sizes, GPU configuration, and other parameters.
/// This requires performing some computations on host CPU, which could take
/// tens of microseconds. To overcome this overhead, it's recommended to query
/// the heristic once using cublasLtMatmulAlgoGetHeuristic() and then reuse
/// result for subsequent computations using cublasLtMatmul().
//------------------------------------------------------------------------------
class LtHeuristic
{
  public:

    LtHeuristic():
      heuristic_{},
      number_of_algorithms_(0)
    {}

    ~LtHeuristic() = default;

    //--------------------------------------------------------------------------
    /// \param [in] requested_algorithm_count - Size of the
    /// heuristicResultsArray, which in our case is named heuristic_. This is
    /// the requested max number of algorithms to return.
    /// Default value from
    /// https://github.com/karpathy/llm.c/blob/master/llmc/matmul.cuh#L205-L206
    /// \param [out] heuristic_ - Array containing algorithm herustics and
    /// associated runtime characteristics, returned by this function, in order
    /// of increasing estimated compute time.
    /// \param [out] number_of_algorithms_ - Number of algorithms returned by
    /// this function, cublasLtMatmulAlgoGetHeuristic().
    //--------------------------------------------------------------------------
    bool get_heuristic(
      cuBLASWrappers::LibraryContextHandle& library_context_handle,
      LtDescriptor& descriptor,
      LtLayouts& layouts,
      LtPreference& preference,
      const int requested_algorithm_count=1);

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/cuda/cublas/index.html?highlight=CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES#cublasltmatmulheuristicresult-t
    /// 3.3.10. cublasLtMatmulHeuristicResult_t
    /// cublasLtHeuristicResult_t is a descriptor that holds the configured
    /// matrix multiplication algorithm descriptor and its runtime properties.
    //--------------------------------------------------------------------------
    cublasLtMatmulHeuristicResult_t heuristic_;
    int number_of_algorithms_;
};

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers

#endif // CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_LT_HEURISTIC_RESULT_H