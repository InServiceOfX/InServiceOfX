ENABLE_TESTING()

ADD_EXECUTABLE(IntegrationTestsCheck
  Configuration/GPUConfiguration_tests.cpp
  Utilities/DeviceManagement/DeviceProperties_tests.cu
  Utilities/DeviceManagement/GetAndSetGPUDevices_tests.cu
  )

TARGET_LINK_LIBRARIES(IntegrationTestsCheck
  Configuration
  Utilities
  gmock_main
  gtest_main
  )

INCLUDE(GoogleTest)
gtest_discover_tests(IntegrationTestsCheck)

SET_TARGET_PROPERTIES(IntegrationTestsCheck
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")