SET(UTILITIES_CU_SOURCES
  DeviceManagement/DeviceProperties.cu
  DeviceManagement/GetAndSetGPUDevices.cu
  ErrorHandling/HandleUnsuccessfulCUDACall.cu
  ErrorHandling/HandleUnsuccessfulCuDNNCall.cu
  )

ADD_LIBRARY(MoreCUDAUtilities
  ${UTILITIES_CU_SOURCES}
  )

# CMAKE_CURRENT_SOURCE_DIR is always the directory of this CMakeLists.txt
# (MoreCUDA/Source/Utilities), so .. is always MoreCUDA/Source/ regardless of
# which project pulls this subdirectory in. PUBLIC propagates the include path
# to any target that links against MoreCUDAUtilities.
TARGET_INCLUDE_DIRECTORIES(MoreCUDAUtilities PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)