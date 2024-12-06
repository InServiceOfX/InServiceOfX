ADD_LIBRARY(cuBLASWrappers
  LibraryContextHandle.cu
  MatrixMultiplication/LtDescriptor.cu
  MatrixMultiplication/LtHeuristic.cu
  MatrixMultiplication/LtLayouts.cu
  MatrixMultiplication/LtSetDescriptorAttributes.cu
  MatrixMultiplication/LtPreference.cu
  MatrixMultiplication/Workspace.cu
  )

TARGET_LINK_LIBRARIES(cuBLASWrappers
  CUDA::cublasLt)
