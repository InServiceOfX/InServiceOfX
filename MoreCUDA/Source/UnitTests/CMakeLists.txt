# Dependencies
# Find a boost install with the libraries unit_test_framework.
# Boost filesystem needs boost system.
FIND_PACKAGE(Boost COMPONENTS filesystem system unit_test_framework REQUIRED)

# Build unit tests
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)

ADD_EXECUTABLE(Check
  Utilities/CaptureCerr.cpp
  Utilities/CaptureCerr_tests.cpp
  Utilities/ErrorHandling/HandleUnsuccessfulCUDACall_tests.cu
  Main.cu
  )

TARGET_LINK_LIBRARIES(Check
  Utilities
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
)

SET_TARGET_PROPERTIES(Check
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")
