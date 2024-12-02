#ifndef NVIDIA_TOOLS_EXTENSION_NVTX_RANGE_H
#define NVIDIA_TOOLS_EXTENSION_NVTX_RANGE_H

#include <string>

namespace NVIDIAToolsExtension
{

//------------------------------------------------------------------------------
/// https://docs.nvidia.com/cuda/profiler-users-guide/#remote-profiling
/// 5. NVIDIA Tools Extension
/// NVIDIA Tools Extension (NVTX) is a C-based Application Programming Interface
/// (API) for annotating events, code ranges, and resources in your
/// applications.
///
/// The NVTX API provides 2 core services:
/// 1. Tracing of CPU events and time ranges.
/// 2. Naming of OS and CUDA resources.
//------------------------------------------------------------------------------

class NVTXRange
{
  public:

    NVTXRange(const char* message);
    NVTXRange(const std::string& message, const int number);
    ~NVTXRange();
};

} // namespace NVIDIAToolsExtension

#define NVTX_RANGE_FN() NVTXRange nvtx_range(__FUNCTION__)
#define NVTX_RANGE_STR(message) NVTXRange nvtx_range(message, __LINE__)

#endif // NVIDIA_TOOLS_EXTENSION_NVTX_RANGE_H