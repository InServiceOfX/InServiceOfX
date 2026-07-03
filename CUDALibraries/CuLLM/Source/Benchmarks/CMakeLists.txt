SET(BENCHMARK_CU_SOURCES
  Transformer/Attention/attention_io_benchmark.cu
  )

ADD_EXECUTABLE(AttentionIOBenchmark
  ${BENCHMARK_CU_SOURCES}
  )

TARGET_LINK_LIBRARIES(AttentionIOBenchmark
  MoreCUDAUtilities
  )

SET_TARGET_PROPERTIES(AttentionIOBenchmark
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

SET_TARGET_PROPERTIES(AttentionIOBenchmark
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")

ADD_EXECUTABLE(AttentionReferenceDump
  Transformer/Attention/attention_reference_dump.cu
  )

TARGET_LINK_LIBRARIES(AttentionReferenceDump
  MoreCUDAUtilities
  )

SET_TARGET_PROPERTIES(AttentionReferenceDump
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

SET_TARGET_PROPERTIES(AttentionReferenceDump
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")

ADD_EXECUTABLE(LinearMapGemmBenchmark
  Transformer/MultiHeadAttention/linear_map_gemm_benchmark.cu
  )

TARGET_LINK_LIBRARIES(LinearMapGemmBenchmark
  MoreCUDAUtilities
  cuBLASWrappers
  StreamManagement
  )

SET_TARGET_PROPERTIES(LinearMapGemmBenchmark
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

SET_TARGET_PROPERTIES(LinearMapGemmBenchmark
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")

ADD_EXECUTABLE(WarpAttentionBenchmark
  Transformer/Attention/warp_attention_benchmark.cu
  )

TARGET_LINK_LIBRARIES(WarpAttentionBenchmark
  MoreCUDAUtilities
  )

SET_TARGET_PROPERTIES(WarpAttentionBenchmark
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

SET_TARGET_PROPERTIES(WarpAttentionBenchmark
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")
