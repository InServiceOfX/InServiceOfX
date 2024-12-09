#ifndef CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_LT_MATRIX_MULTIPLICATION_H
#define CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_LT_MATRIX_MULTIPLICATION_H

#include "cuBLASWrappers/LibraryContextHandle.h"
#include "cuBLASWrappers/MatrixMultiplication/LtDescriptor.h"
#include "cuBLASWrappers/MatrixMultiplication/LtHeuristic.h"
#include "cuBLASWrappers/MatrixMultiplication/LtLayouts.h"
#include "cuBLASWrappers/MatrixMultiplication/Setup.h"
#include "cuBLASWrappers/MatrixMultiplication/Workspace.h"
#include "DataStructures/Array.h"
#include "StreamManagement/Stream.h"

#include <iostream>

namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

template<typename T>
class LtMatrixMultiplication
{
  public:

    LtMatrixMultiplication(
      const T alpha=1.0,
      // If beta is anything but 0 and the bias is not setup, such as the
      // CUBLAS_MATMUL_DESC_BIAS_DATA_TYPE, or if a nullptr was passed into the
      // bias parameter (i.e. matrix C), then matrix multiplication will fail.
      // This was non-obvious to me, so I'll repeat below.
      const T beta=0.0):
      alpha_{alpha},
      beta_{beta}
    {}

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/cuda/cublas/index.html?highlight=CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES#cublasltmatmul
    /// 3.4.17. cublasLtMatmul()
    /// cublasStatus_t cublasLtMatmul(
    ///   cublasLtHandle_t lightHandle,
    ///   const cublasLtMatmulDesc_t computeDesc,
    ///   const void *alpha,
    ///   const void *A,
    ///   cublasMatrixLayout_t Adesc,
    ///   const void *B,
    ///   cublasMatrixLayout_t Bdesc,
    ///   const void *beta,
    ///   const void *C,
    ///   cublasMatrixLayout_t Cdesc,
    ///   void *D,
    ///   cublasMatrixLayout_t Ddesc,
    ///   const cublasLtMatmulAlgo_t* algo
    ///   ..)
    ///
    /// D = alpha*(A*B) + beta*(C)
    //--------------------------------------------------------------------------
    bool operator()(
      cuBLASWrappers::LibraryContextHandle& handle,
      LtDescriptor& descriptor,
      LtLayouts& layouts,
      LtHeuristic& heuristic,
      StreamManagement::Stream& stream,
      Workspace& workspace,
      const T* A,
      const T* B,
      const T* C,
      T* D)
    {
      return handle_matrix_multiplication(
        cublasLtMatmul(
          handle.handle_,
          descriptor.descriptor_,
          &alpha_,
          A,
          layouts.A_layout_,
          B,
          layouts.B_layout_,
          &beta_,
          C,
          layouts.C_layout_,
          D,
          layouts.D_layout_,
          &heuristic.heuristic_.algo,
          workspace.workspace_,
          workspace.workspace_size_in_bytes_,
          stream.stream_));
    }

    inline void set_alpha(const T alpha)
    {
      alpha_ = alpha;
    }

    inline void set_beta(const T beta)
    {
      beta_ = beta;
    }

  private:

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/cuda/cublas/index.html?highlight=CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES#cublasltmatmul
    /// 3.4.17. cublasLtMatmul()
    //--------------------------------------------------------------------------
    bool handle_matrix_multiplication(const cublasStatus_t status)
    {
      if (status == CUBLAS_STATUS_SUCCESS)
      {
        return true;
      }
      else if (status == CUBLAS_STATUS_NOT_INITIALIZED)
      {
        std::cerr << "cuBLASLt handle hasn't been initialized.\n";
      }
      else if (status == CUBLAS_STATUS_INVALID_VALUE)
      {
        static constexpr const char* error_message_1 {
          "Either parameters are unexpected NULL, in conflict, or in an \n"};
        static constexpr const char* error_message_2 {
          "impossible configuration.\n"};

        std::cerr << error_message_1 << error_message_2 << "\n";
      }
      else if (status == CUBLAS_STATUS_NOT_SUPPORTED)
      {
        static constexpr const char* error_message_1 {
          "Current implementation on selected device doesn't support the \n"};
        static constexpr const char* error_message_2 {
          "configured operation.\n"};

        std::cerr << error_message_1 << error_message_2 << "\n";
      }
      else if (status == CUBLAS_STATUS_ARCH_MISMATCH)
      {
        std::cerr << "Configured operation can't be run using selected device.\n";
      }
      else if (status == CUBLAS_STATUS_EXECUTION_FAILED)
      {
        std::cerr << "CUDA reported an execution error from the device.\n";
      }
      else
      {
        std::cerr << "Failed to launch cuBLASLt matrix multiplication.\n";
      }

      return false;
    }

    T alpha_;
    // If beta is anything but 0 and the bias is not setup, such as the
    // CUBLAS_MATMUL_DESC_BIAS_DATA_TYPE, or if a nullptr was passed into the
    // bias parameter (i.e. matrix C), then matrix multiplication will fail.
    // This was non-obvious to me.
    T beta_;
};

template <typename T>
bool matrix_multiply(
  cuBLASWrappers::LibraryContextHandle& handle,
  StreamManagement::Stream& stream,
  Setup<T>& setup,
  LtMatrixMultiplication<T>& matrix_multiplication,
  DataStructures::Array<T>& A,
  DataStructures::Array<T>& B,
  DataStructures::Array<T>& D)
{
  return matrix_multiplication(
    handle,
    setup.descriptor_,
    setup.layouts_,
    setup.heuristic_,
    stream,
    setup.workspace_,
    A.elements_,
    B.elements_,
    nullptr,
    D.elements_);
}

template <typename T>
bool general_matrix_multiply(
  cuBLASWrappers::LibraryContextHandle& handle,
  StreamManagement::Stream& stream,
  Setup<T>& setup,
  LtMatrixMultiplication<T>& matrix_multiplication,
  DataStructures::Array<T>& A,
  DataStructures::Array<T>& B,
  DataStructures::Array<T>& D)
{
  return matrix_multiplication(
    handle,
    setup.descriptor_,
    setup.layouts_,
    setup.heuristic_,
    stream,
    setup.workspace_,
    A.elements_,
    B.elements_,
    D.elements_,
    D.elements_);
}

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers

#endif // CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_LT_MATRIX_MULTIPLICATION_H