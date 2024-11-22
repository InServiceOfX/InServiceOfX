ENABLE_TESTING()

ADD_EXECUTABLE(Check
  ArbitraryIPAddressSetup.cpp
  IPC/UDPSocket_tests.cpp
  IPC/UDP/RunningUDPTransceiver_tests.cpp
  Utilities/Parsers/parse_simple_map_tests.cpp
  )

TARGET_LINK_LIBRARIES(Check
  IPC
  Utilities
  gmock_main
  gtest_main
  )

INCLUDE(GoogleTest)
gtest_discover_tests(Check)

SET_TARGET_PROPERTIES(Check
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")
