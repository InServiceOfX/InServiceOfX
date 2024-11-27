ENABLE_TESTING()

ADD_EXECUTABLE(Check
  DataStructures/Array_tests.cu
  GPUPuzzles_tests.cu
  Numerics/Constants/get_infinity_tests.cu
  Operations/Arithmetic_tests.cu
  Operations/Convolution1D_tests.cu
  Operations/dot_product_tests.cu
  Utilities/CaptureCerr.cpp
  Utilities/CaptureCerr_tests.cpp
  Utilities/ErrorHandling/HandleUnsuccessfulCUDACall_tests.cu
  Utilities/arange_tests.cpp
  )

TARGET_LINK_LIBRARIES(Check
  Utilities
  gmock_main
  gtest_main
)

INCLUDE(GoogleTest)
gtest_discover_tests(Check)

# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_SEPARABLE_COMPILATION.html
# CUDA_SEPARABLE_COMPILATION enables separate compilation for all CUDA files
# for the given target.
# It should allow us to compile multiple CUDA source files into separate device
# object files and then link them together into a single executable or library.
# It should be necessary when using device code linking (e.g. __device__ or
# __global__ functions in different translation units).
SET_TARGET_PROPERTIES(Check
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

SET_TARGET_PROPERTIES(Check
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")
