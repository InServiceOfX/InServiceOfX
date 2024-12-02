#include "cuBLASWrappers/MatrixMultiplication/cuBLASLtDescriptor.h"

#include <cublasLt.h>
#include <stdexcept>

namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

cuBLASLtDescriptor::cuBLASLtDescriptor():
  descriptor_{}
{
  create_descriptor(get_compute_parameters<float>());
}

cuBLASLtDescriptor::cuBLASLtDescriptor(
	const ComputeParameters compute_parameters):
  descriptor_{}
{
  create_descriptor(compute_parameters);
}

cuBLASLtDescriptor::~cuBLASLtDescriptor()
{
  destroy_descriptor();
}

bool cuBLASLtDescriptor::create_descriptor(
  const ComputeParameters compute_parameters)
{
	// https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldesccreate
	// cublasStatus_t cublasLtMatmulDescCreate(cublashLtMatmulDesc_t *matmulDesc,
  //   cublasComputeType_t computeType,
	//   cudaDataType_t scaleType)
  const cublasStatus_t status {
    cublasLtMatmulDescCreate(
      &descriptor_,
      compute_parameters.compute_precision_mode_,
      compute_parameters.data_type_)};

  if (status != CUBLAS_STATUS_SUCCESS)
  {
		if (status == CUBLAS_STATUS_ALLOC_FAILED)
		{
			throw std::runtime_error("Memory could not be allocated.");
		}

    throw std::runtime_error("Failed to create cuBLASLt descriptor");
  }

  return status == CUBLAS_STATUS_SUCCESS;
}

bool cuBLASLtDescriptor::destroy_descriptor()
{
  return cublasLtMatmulDescDestroy(descriptor_) == CUBLAS_STATUS_SUCCESS;
}

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers
