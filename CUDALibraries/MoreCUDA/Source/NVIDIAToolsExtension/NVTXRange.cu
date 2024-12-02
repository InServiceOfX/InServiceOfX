#include "NVIDIAToolsExtension/NVTXRange.h"

#include <nvtx3/nvToolsExtCudaRt.h>
#include <string>

namespace NVIDIAToolsExtension
{

NVTXRange::NVTXRange(const char* message)
{
  nvtxRangePush(message);
}

NVTXRange::NVTXRange(const std::string& message, const int number)
{
  std::string range_string {message + " " + std::to_string(number)};

  nvtxRangePush(range_string.c_str());
}

NVTXRange::~NVTXRange()
{
  nvtxRangePop();
}

} // namespace NVIDIAToolsExtension
