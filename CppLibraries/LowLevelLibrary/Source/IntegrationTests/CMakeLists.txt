ENABLE_TESTING()

ADD_EXECUTABLE(IntegrationTestsCheck
  IPC/UDPSocket_tests.cpp
  IPC/UDP/RunningUDPTransceiver_tests.cpp
  IPC/UDP/RunningTwoUDPTransceivers_tests.cpp
  ../UnitTests/ArbitraryIPAddressSetup.cpp
  )

TARGET_LINK_LIBRARIES(IntegrationTestsCheck
  IPC
  Utilities
  gmock_main
  gtest_main
  )

INCLUDE(GoogleTest)
gtest_discover_tests(IntegrationTestsCheck)

SET_TARGET_PROPERTIES(IntegrationTestsCheck
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")

# Executables

ADD_EXECUTABLE(udp_socket_sender
  IPC/udp_socket_sender.cpp
	)

TARGET_LINK_LIBRARIES(udp_socket_sender PRIVATE
  IPC
  Utilities
	)

SET_TARGET_PROPERTIES(udp_socket_sender
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")

ADD_EXECUTABLE(running_udp_transceiver
  IPC/UDP/running_udp_transceiver.cpp
  ../UnitTests/ArbitraryIPAddressSetup.cpp
	)

TARGET_LINK_LIBRARIES(running_udp_transceiver PRIVATE
  IPC
  Utilities
	)

SET_TARGET_PROPERTIES(running_udp_transceiver
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")

ADD_EXECUTABLE(udp_socket_receiver
  IPC/udp_socket_receiver.cpp
)

TARGET_LINK_LIBRARIES(udp_socket_receiver PRIVATE
  IPC
  Utilities
  )

SET_TARGET_PROPERTIES(udp_socket_receiver
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")