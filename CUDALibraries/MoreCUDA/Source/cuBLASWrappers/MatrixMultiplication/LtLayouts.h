#ifndef CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_LT_LAYOUTS_H
#define CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_LT_LAYOUTS_H

#include "cuBLASWrappers/get_data_precision.h"

#include <cublasLt.h>
#include <cstdint>
#include <optional>
#include <tuple>

namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

//------------------------------------------------------------------------------
/// Recall from
/// https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
/// That matrix multiplication computes
/// D = alpha * (A * B) + beta * C
//------------------------------------------------------------------------------

class LtLayouts
{
  public:

    LtLayouts():
      A_layout_{},
      B_layout_{},
      C_layout_{},
      D_layout_{},
      m_{},
      n_{},
      k_{},
      batch_count_{},
      A_strided_batch_offset_{},
      B_strided_batch_offset_{},
      output_strided_batch_offset_{},
      memory_order_{CUBLASLT_ORDER_COL}
    {}

    LtLayouts(
      const uint64_t m,
      const uint64_t n,
      const uint64_t k):
      A_layout_{},
      B_layout_{},
      C_layout_{},
      D_layout_{},
      m_{m},
      n_{n},
      k_{k},
      batch_count_{},
      A_strided_batch_offset_{},
      B_strided_batch_offset_{},
      output_strided_batch_offset_{},
      memory_order_{CUBLASLT_ORDER_COL}
    {}

    ~LtLayouts();

    void set_dimensions(const uint64_t m, const uint64_t n, const uint64_t k);

    template <typename T>
    bool create_ABD_layouts(
      const bool is_transpose_A=false,
      const bool is_transpose_B=false)
    {
      if (is_transpose_A)
      {
        const cublasStatus_t status_A {
          cublasLtMatrixLayoutCreate(
            &A_layout_,
            get_data_precision<T>(),
            k_,
            m_,
            k_)};

        if (!handle_create_layout_status(status_A, 'A'))
        {
          return false;
        }
      }
      else
      {
        const cublasStatus_t status_A {
          cublasLtMatrixLayoutCreate(
            &A_layout_,
            get_data_precision<T>(),
            m_,
            k_,
            m_)};

        if (!handle_create_layout_status(status_A, 'A'))
        {
          return false;
        }
      }

      if (is_transpose_B)
      {
        const cublasStatus_t status_B {
          cublasLtMatrixLayoutCreate(
            &B_layout_,
            get_data_precision<T>(),
            n_,
            k_,
            n_)};

        if (!handle_create_layout_status(status_B, 'B'))
        {
          return false;
        }
      }
      else
      {
        const cublasStatus_t status_B {
          cublasLtMatrixLayoutCreate(
            &B_layout_,
            get_data_precision<T>(),
            k_,
            n_,
            k_)};

        if (!handle_create_layout_status(status_B, 'B'))
        {
          return false;
        }
      }

      return handle_create_layout_status(
        cublasLtMatrixLayoutCreate(
          &D_layout_,
          get_data_precision<T>(),
          m_,
          n_,
          m_),
        'D');
    }

    //--------------------------------------------------------------------------
    /// "cuBLASTLt requires C in FFP8 mode to be BF16 or FP32...(sigh)"
    //--------------------------------------------------------------------------
    template <typename T>
    bool create_C_layout()
    {
      const cublasStatus_t status {
        cublasLtMatrixLayoutCreate(
          &C_layout_,
          (sizeof(T) == 1) ? CUDA_R_16BF : get_data_precision<T>(),
          m_,
          n_,
          m_)};

      return handle_create_layout_status(status, 'C');
    }

    bool set_batch_count_and_strided_offsets(
      const int32_t batch_count,
      const int64_t A_strided_batch_offset=0,
      const int64_t B_strided_batch_offset=0,
      const int64_t output_strided_batch_offset=0);

    std::optional<std::tuple<int32_t, uint64_t>> get_batch_count(
      const char matrix_name) const;

    std::optional<std::tuple<int64_t, uint64_t>> get_strided_batch_offset(
      const char matrix_name) const;

    bool set_memory_order(
      const char matrix_name,
      cublasLtOrder_t data_ordering=CUBLASLT_ORDER_COL);

    std::optional<std::tuple<cublasLtOrder_t, uint64_t>> get_memory_order(
      const char matrix_name);

    cublasLtMatrixLayout_t A_layout_;
    cublasLtMatrixLayout_t B_layout_;
    cublasLtMatrixLayout_t C_layout_;
    cublasLtMatrixLayout_t D_layout_;

  protected:

    bool destroy_layouts();

    bool handle_create_layout_status(
      const cublasStatus_t status,
      const char matrix_name);

    static bool handle_set_attribute(
      const cublasStatus_t status,
      cublasLtMatrixLayoutAttribute_t attribute);

    static bool handle_get_attribute(
      const cublasStatus_t status,
      cublasLtMatrixLayoutAttribute_t attribute);

    // Number of rows of matrix A, the "left matrix" for matrix multiplication.
    uint64_t m_;
    // Number of columns of matrix B, the "right matrix" for matrix
    // multiplication.
    uint64_t n_;
    // Number of columns of A = number of rows of B (matrix multiplication
    // requires this).
    uint64_t k_;

    int32_t batch_count_;

    int64_t A_strided_batch_offset_;
    int64_t B_strided_batch_offset_;
    int64_t output_strided_batch_offset_;

    cublasLtOrder_t memory_order_;
};

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers

#endif // CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_LT_LAYOUTS_H
