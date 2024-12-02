ADD_LIBRARY(cuBLASWrappers
  LibraryContextHandle.cu
  MatrixMultiplication/cuBLASLtDescriptor.cu
  MatrixMultiplication/cuBLASLtLayouts.cu
  MatrixMultiplication/cuBLASLtSetDescriptorAttributes.cu
  )

TARGET_LINK_LIBRARIES(cuBLASWrappers
  CUDA::cublasLt)
