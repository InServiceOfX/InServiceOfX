SET(UTILITIES_CU_SOURCES
  DeviceManagement/DeviceProperties.cu
  DeviceManagement/GetAndSetGPUDevices.cu
  ErrorHandling/HandleUnsuccessfulCUDACall.cu
  ErrorHandling/HandleUnsuccessfulCuDNNCall.cu
  )

ADD_LIBRARY(Utilities
  ${UTILITIES_CU_SOURCES}
  )