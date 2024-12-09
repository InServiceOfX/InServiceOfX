#ifndef CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_LT_SET_DESCRIPTOR_ATTRIBUTES_H
#define CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_LT_SET_DESCRIPTOR_ATTRIBUTES_H

#include "cuBLASWrappers/get_data_precision.h"

#include <cstdint>
#include <cublasLt.h>
#include <optional>
#include <utility>

namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

class LtSetDescriptorAttributes
{
  public:

    LtSetDescriptorAttributes():
      no_transpose_{CUBLAS_OP_N},
      transpose_{CUBLAS_OP_T},
      gelu_leading_dimension_{},
      // https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescattributes-t
      // Default value is CUBLASLT_EPILOGUE_DEFAULT.
      epilogue_{},
      bias_data_type_{},
      // Set scale type to FP32 (needs to be FP16 if and only if using
      // CUBLAS_COMPUTE_16F, so it's FP32 even for FP8!). See
      // https://github.com/karpathy/llm.c/blob/master/llmc/matmul.cuh#L200-L201
      scale_type_{CUDA_R_32F}
    {}

    ~LtSetDescriptorAttributes() = default;

    //--------------------------------------------------------------------------
    /// \param [in] is_transpose - If true, matrix A is transposed.
    //--------------------------------------------------------------------------
    bool set_transpose_on_A(
      cublasLtMatmulDesc_t matmul_descriptor,
      const bool is_transpose=false);
    bool set_transpose_on_B(
      cublasLtMatmulDesc_t matmul_descriptor,
      const bool is_transpose=false);

    std::optional<std::pair<int32_t, uint64_t>> get_transpose_operation_on_A(
      cublasLtMatmulDesc_t matmul_descriptor);
    std::optional<std::pair<int32_t, uint64_t>> get_transpose_operation_on_B(
      cublasLtMatmulDesc_t matmul_descriptor);

    //--------------------------------------------------------------------------
    /// See
    /// https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldesc-t
    /// CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD - Leading dimension for epilogue
    /// auxiliary buffer.
    /// GELU input matrix leading dimension in elements when
    /// CUBLASLT_EPILOGUE_GELU_AUX_BIAS, ... epilogue used. Must be divisible by
    /// 8 and be no less than number of rows in output matrix.
    /// \param [in] m - typically number of rows in output matrix.
    //--------------------------------------------------------------------------
    bool set_gelu_epilogue_auxiliary_leading_dimension(
      cublasLtMatmulDesc_t matmul_descriptor,
      const int64_t m);

    //--------------------------------------------------------------------------
    /// CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER - Pointer for epilogue
    /// auxiliary buffer.
    //--------------------------------------------------------------------------
    template <typename T>
    inline bool set_gelu_epilogue_auxiliary_pointer(
      cublasLtMatmulDesc_t matmul_descriptor,
      T* pre_gelu)
    {
      return handle_set_descriptor_status(
        cublasLtMatmulDescSetAttribute(
          matmul_descriptor,
          CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
          &pre_gelu,
          sizeof(pre_gelu)));
    }

    //--------------------------------------------------------------------------
    /// Expected behavior:
    /// We shouldn't have any backward matrix multiplications that use both GELU
    /// and bias.
    /// If we have backward matrix multiplication and uses GELU but no bias,
    /// then set to CUBLASLT_EPILOGUE_DGELU.
    //--------------------------------------------------------------------------
    void set_epilogue(
      const bool has_gelu,
      const bool is_backward,
      const bool has_bias);

    bool set_epilogue_function(cublasLtMatmulDesc_t matmul_descriptor);

    static uint64_t get_bias_size(
      const uint64_t m,
      const uint64_t n,
      const bool has_gelu=false,
      const bool is_backward=false);

    template <typename T>
    bool set_bias(cublasLtMatmulDesc_t matmul_descriptor, const T* bias)
    {
      // TODO: Confirm or fix, cuBLASLt requires bias in FP8 mode to be BF16
      // "... (sigh)"
      // See https://github.com/karpathy/llm.c/blob/master/llmc/matmul.cuh#L194
      bias_data_type_ = (sizeof(T) == 1) ?
        CUDA_R_16BF : get_data_precision<T>();

      // CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE - Type of bias or bias gradient
      // vector in the device memory. Bias case: see CUBLASLT_EPILOGUE_BIAS. If
      // unset (or set to default value of -1), bias vector elements are same
      // type as elements of output matrix (Dtype), with exceptions.
      const bool is_bias_data_type_set {
        handle_set_descriptor_status(
          cublasLtMatmulDescSetAttribute(
            matmul_descriptor,
            CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
            &bias_data_type_,
            sizeof(bias_data_type_)))};

      if (!is_bias_data_type_set)
      {
        return false;
      }

      return handle_set_descriptor_status(
        cublasLtMatmulDescSetAttribute(
          matmul_descriptor,
          CUBLASLT_MATMUL_DESC_BIAS_POINTER,
          &bias,
          sizeof(bias)));
    }

    bool set_scale_type(
      cublasLtMatmulDesc_t matmul_descriptor,
      const cublasDataType_t scale_type = CUDA_R_32F);

  protected:

    bool handle_set_descriptor_status(const cublasStatus_t status);
    bool handle_get_attribute(const cublasStatus_t status);

    static cublasLtEpilogue_t get_epilogue_postprocessing_options(
      const bool has_gelu,
      const bool is_backward,
      const bool has_bias);

    cublasOperation_t no_transpose_;
    cublasOperation_t transpose_;
    int64_t gelu_leading_dimension_;
    // https://docs.nvidia.com/cuda/cublas/#cublasltepilogue-t
    // 3.3.2. cublasLtEpilogue_t
    // cublasLtEpilogue_t is an enum type to set postprocessing options for
    // epilogue.
    cublasLtEpilogue_t epilogue_;
    cublasDataType_t bias_data_type_;
    cublasDataType_t scale_type_;
};

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers

#endif // CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_LT_SET_DESCRIPTOR_ATTRIBUTES_H
