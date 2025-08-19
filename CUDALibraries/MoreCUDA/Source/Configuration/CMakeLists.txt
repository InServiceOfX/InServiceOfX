SET(COMMON_DIR ${CMAKE_SOURCE_DIR}/../../../CppLibraries/Common)

IF (NOT EXISTS ${COMMON_DIR})
  MESSAGE(FATAL_ERROR "Common directory not found: ${COMMON_DIR}")
ENDIF()

ADD_LIBRARY(Configuration
  GPUConfiguration.cpp
)

TARGET_INCLUDE_DIRECTORIES(Configuration
  PUBLIC
  ${COMMON_DIR}/Source
)
