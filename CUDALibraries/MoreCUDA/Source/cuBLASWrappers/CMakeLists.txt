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

# CMAKE_CURRENT_SOURCE_DIR is always the directory of this CMakeLists.txt
# (MoreCUDA/Source/cuBLASWrappers), so .. is always MoreCUDA/Source/ regardless
# of which project pulls this subdirectory in. PUBLIC propagates the include
# path to any target that links against cuBLASWrappers, and is also required
# to compile cuBLASWrappers' own .cu files (e.g. #include
# "cuBLASWrappers/LibraryContextHandle.h") when this directory is added via
# ADD_SUBDIRECTORY from a project other than MoreCUDA itself, since only
# MoreCUDA/Source/CMakeLists.txt's global INCLUDE_DIRECTORIES(BEFORE
# ${CMAKE_SOURCE_DIR}) covered this before.
TARGET_INCLUDE_DIRECTORIES(cuBLASWrappers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
