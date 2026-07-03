ADD_LIBRARY(StreamManagement
  Stream.cu
  )

# CMAKE_CURRENT_SOURCE_DIR is always the directory of this CMakeLists.txt
# (MoreCUDA/Source/StreamManagement), so .. is always MoreCUDA/Source/
# regardless of which project pulls this subdirectory in. PUBLIC propagates
# the include path to any target that links against StreamManagement, and is
# also required to compile Stream.cu itself (#include
# "StreamManagement/Stream.h", "Utilities/ErrorHandling/...") when this
# directory is added via ADD_SUBDIRECTORY from a project other than MoreCUDA.
TARGET_INCLUDE_DIRECTORIES(StreamManagement PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
