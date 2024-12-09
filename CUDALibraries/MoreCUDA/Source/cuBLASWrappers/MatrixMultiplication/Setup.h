#ifndef CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_SETUP_H
#define CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_SETUP_H

#include "cuBLASWrappers/LibraryContextHandle.h"
#include "cuBLASWrappers/MatrixMultiplication/LtDescriptor.h"
#include "cuBLASWrappers/MatrixMultiplication/LtHeuristic.h"
#include "cuBLASWrappers/MatrixMultiplication/LtLayouts.h"
#include "cuBLASWrappers/MatrixMultiplication/LtPreference.h"
#include "cuBLASWrappers/MatrixMultiplication/LtSetDescriptorAttributes.h"
#include "cuBLASWrappers/MatrixMultiplication/Workspace.h"

#include <cstdint>
#include <optional>

namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

template<typename T>
class Setup
{
  public:

    using LtDescriptor = cuBLASWrappers::MatrixMultiplication::LtDescriptor;
    using LtHeuristic = cuBLASWrappers::MatrixMultiplication::LtHeuristic;
    using LtLayouts = cuBLASWrappers::MatrixMultiplication::LtLayouts;
    using LtPreference = cuBLASWrappers::MatrixMultiplication::LtPreference;
    using LtSetDescriptorAttributes =
      cuBLASWrappers::MatrixMultiplication::LtSetDescriptorAttributes;
    using Workspace = cuBLASWrappers::MatrixMultiplication::Workspace;

    Setup():
      workspace_{},
      descriptor_{get_compute_parameters<T>()},
      set_descriptor_attributes_{},
      layouts_{},
      preference_{},
      heuristic_{},
      M_{0},
      N_{0},
      K_{0}
    {}

    Setup(const uint32_t M, const uint32_t N, const uint32_t K):
      workspace_{},
      descriptor_{get_compute_parameters<T>()},
      set_descriptor_attributes_{},
      layouts_{M, N, K},
      preference_{},
      heuristic_{},
      M_{M},
      N_{N},
      K_{K}
    {}

    void set_dimensions(const uint32_t M, const uint32_t N, const uint32_t K)
    {
      M_ = M;
      N_ = N;
      K_ = K;
      layouts_.set_dimensions(M, N, K);
    }

    struct BatchCountAndStridedOffsets
    {
      int32_t batch_count_{0};
      int64_t A_strided_batch_offset_{0};
      int64_t B_strided_batch_offset_{0};
      int64_t output_strided_batch_offset_{0};

      void set_strided_offsets(
        const uint32_t M,
        const uint32_t N,
        const uint32_t K)
      {
        A_strided_batch_offset_ = M * K;
        B_strided_batch_offset_ = N * K;
        output_strided_batch_offset_ = M * N;
      }
    };

    bool setup(
      cuBLASWrappers::LibraryContextHandle& handle,
      const bool is_transpose_on_A=false,
      const bool is_transpose_on_B=false,
      const std::optional<BatchCountAndStridedOffsets>
        batch_count_and_strided_offsets=std::nullopt,
      T* bias_pointer=nullptr)
    {
      bool is_success {true};

      is_success = is_success && set_descriptor_attributes_.set_transpose_on_A(
        descriptor_.descriptor_,
        is_transpose_on_A);
      is_success = is_success && set_descriptor_attributes_.set_transpose_on_B(
        descriptor_.descriptor_,
        is_transpose_on_B);

      is_success = is_success && layouts_.create_ABD_layouts<T>(
        is_transpose_on_A,
        is_transpose_on_B);
      is_success = is_success && layouts_.create_C_layout<T>();

      if (batch_count_and_strided_offsets)
      {
        is_success = is_success && layouts_.set_batch_count_and_strided_offsets(
          batch_count_and_strided_offsets->batch_count_,
          batch_count_and_strided_offsets->A_strided_batch_offset_,
          batch_count_and_strided_offsets->B_strided_batch_offset_,
          batch_count_and_strided_offsets->output_strided_batch_offset_);
      }

      is_success = is_success && preference_.set_max_workspace_memory(
        workspace_.workspace_size_in_bytes_);

      if (bias_pointer == nullptr)
      {
        set_descriptor_attributes_.set_epilogue(false, false, false);
      }
      else
      {
        set_descriptor_attributes_.set_epilogue(false, false, true);
      }

      is_success = is_success &&
        set_descriptor_attributes_.set_epilogue_function(
          descriptor_.descriptor_);

      if (bias_pointer != nullptr)
      {
        is_success = is_success && set_descriptor_attributes_.set_bias(
          descriptor_.descriptor_,
          bias_pointer);
      }

      is_success = is_success && set_descriptor_attributes_.set_scale_type(
        descriptor_.descriptor_);

      is_success = is_success && heuristic_.get_heuristic(
        handle,
        descriptor_,
        layouts_,
        preference_);

      return is_success;
    }

    bool setup_with_gelu(
      cuBLASWrappers::LibraryContextHandle& handle,
      T* pre_gelu_pointer,
      const bool is_transpose_on_A=false,
      const bool is_transpose_on_B=false,
      const std::optional<BatchCountAndStridedOffsets>
        batch_count_and_strided_offsets=std::nullopt,
      T* bias_pointer=nullptr)
    {
      bool is_success {true};

      is_success = is_success && set_descriptor_attributes_.set_transpose_on_A(
        descriptor_.descriptor_,
        is_transpose_on_A);
      is_success = is_success && set_descriptor_attributes_.set_transpose_on_B(
        descriptor_.descriptor_,
        is_transpose_on_B);

      is_success = is_success && layouts_.create_ABD_layouts<T>(
        is_transpose_on_A,
        is_transpose_on_B);
      is_success = is_success && layouts_.create_C_layout<T>();

      if (batch_count_and_strided_offsets)
      {
        is_success = is_success && layouts_.set_batch_count_and_strided_offsets(
          batch_count_and_strided_offsets->batch_count_,
          batch_count_and_strided_offsets->A_strided_batch_offset_,
          batch_count_and_strided_offsets->B_strided_batch_offset_,
          batch_count_and_strided_offsets->output_strided_batch_offset_);
      }

      is_success = is_success && preference_.set_max_workspace_memory(
        workspace_.workspace_size_in_bytes_);

      is_success = is_success && set_descriptor_attributes_.set_gelu_epilogue_auxiliary_leading_dimension(
        descriptor_.descriptor_,
        M_);
      is_success = is_success && set_descriptor_attributes_.set_gelu_epilogue_auxiliary_pointer(
        descriptor_.descriptor_,
        pre_gelu_pointer);

      if (bias_pointer == nullptr)
      {
        set_descriptor_attributes_.set_epilogue(true, false, false);
      }
      else
      {
        set_descriptor_attributes_.set_epilogue(true, false, true);
      }

      is_success = is_success &&
        set_descriptor_attributes_.set_epilogue_function(
          descriptor_.descriptor_);

      if (bias_pointer != nullptr)
      {
        is_success = is_success && set_descriptor_attributes_.set_bias(
          descriptor_.descriptor_,
          bias_pointer);
      }

      is_success = is_success && set_descriptor_attributes_.set_scale_type(
        descriptor_.descriptor_);

      is_success = is_success && heuristic_.get_heuristic(
        handle,
        descriptor_,
        layouts_,
        preference_);

      return is_success;
    }


    Workspace workspace_;
    LtDescriptor descriptor_;
    LtSetDescriptorAttributes set_descriptor_attributes_;
    LtLayouts layouts_;
    LtPreference preference_;
    LtHeuristic heuristic_;

  protected:

    uint32_t M_;
    uint32_t N_;
    uint32_t K_;
};

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers

#endif // CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_SETUP_H