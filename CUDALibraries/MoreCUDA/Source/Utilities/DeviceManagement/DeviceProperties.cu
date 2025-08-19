#include "Utilities/ErrorHandling/HandleUnsuccessfulCUDACall.h"
#include "Utilities/DeviceManagement/DeviceProperties.h"

#include <array>
#include <cuda_runtime.h>
#include <iomanip>
#include <sstream>
#include <string>

using Utilities::ErrorHandling::HandleUnsuccessfulCUDACall;

namespace Utilities
{
namespace DeviceManagement
{

DeviceProperties::DeviceProperties():
  device_properties_()
{}

DeviceProperties::DeviceProperties(const cudaDeviceProp device_properties):
    device_properties_(device_properties)
{}

bool DeviceProperties::get_device_properties(const int device_id)
{
  HandleUnsuccessfulCUDACall handle_get_device_properties {
    "Failed to get device properties"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_get_device_properties,
    cudaGetDeviceProperties(&device_properties_, device_id));

  return handle_get_device_properties.is_cuda_success();
}

std::string DeviceProperties::get_compute_capability() const
{
  std::ostringstream oss;
  oss << device_properties_.major << "." << device_properties_.minor;
  return oss.str();
}

std::string DeviceProperties::format_to_bytes(const size_t bytes)
{
  static constexpr std::array<std::string_view, 5> units {
    "B", "KB", "MB", "GB", "TB"};

  int unit_index ={0};
  double size {static_cast<double>(bytes)};

  while (size >= 1024.0 && unit_index < 4) {
    size /= 1024.0;
    unit_index++;
  }

  std::ostringstream oss {};
  oss << std::fixed <<
    std::setprecision(size < 10 ? 2 : (size < 100 ? 1 : 0)) << size << " " <<
    units[unit_index];
  return oss.str();
}

std::string DeviceProperties::format_int3(const int x[3])
{
  std::ostringstream oss {};
  oss << "[" << x[0] << ", " << x[1] << ", " << x[2] << "]";
  return oss.str();
}

} // namespace DeviceManagement
} // namespace Utilities

/*
#include <iomanip>
#include <algorithm>


std::string GPUDeviceProperties::get_summary_string() const
{
    if (!is_initialized_) {
        return "GPU Device Properties: Not initialized";
    }
    
    std::ostringstream oss;
    oss << "=== GPU Device Properties Summary ===\n";
    oss << "Device Name: " << properties_.name << "\n";
    oss << "Compute Capability: " << properties_.major << "." << properties_.minor << "\n";
    oss << "Multiprocessors: " << properties_.multiProcessorCount << "\n";
    oss << "Total Memory: " << format_memory_size(properties_.totalGlobalMem) << "\n";
    oss << "Shared Memory per Block: " << format_memory_size(properties_.sharedMemPerBlock) << "\n";
    oss << "Max Threads per Block: " << properties_.maxThreadsPerBlock << "\n";
    oss << "Max Threads per SM: " << properties_.maxThreadsPerMultiProcessor << "\n";
    oss << "Max Warp Size: " << properties_.warpSize << "\n";
    oss << "Clock Rate: " << properties_.clockRate / 1000 << " MHz\n";
    oss << "Memory Clock Rate: " << properties_.memoryClockRate / 1000 << " MHz\n";
    oss << "Memory Bus Width: " << properties_.memoryBusWidth << " bits\n";
    oss << "L2 Cache Size: " << format_memory_size(properties_.l2CacheSize) << "\n";
    oss << "Unified Memory: " << (supports_unified_memory() ? "Yes" : "No") << "\n";
    oss << "Concurrent Kernels: " << (supports_concurrent_kernels() ? "Yes" : "No") << "\n";
    oss << "Managed Memory: " << (supports_managed_memory() ? "Yes" : "No") << "\n";
    
    return oss.str();
}

std::string GPUDeviceProperties::get_device_name() const
{
    if (!is_initialized_) return "Unknown Device";
    return std::string(properties_.name);
}

std::string GPUDeviceProperties::get_compute_capability() const
{
    if (!is_initialized_) return "Unknown";
    std::ostringstream oss;
    oss << "SM " << properties_.major << "." << properties_.minor;
    return oss.str();
}

std::string GPUDeviceProperties::get_memory_info() const
{
    if (!is_initialized_) return "Unknown";
    
    std::ostringstream oss;
    oss << "Total Global Memory: " << format_memory_size(properties_.totalGlobalMem) << "\n";
    oss << "Shared Memory per Block: " << format_memory_size(properties_.sharedMemPerBlock) << "\n";
    oss << "Shared Memory per SM: " << format_memory_size(properties_.sharedMemPerMultiprocessor) << "\n";
    oss << "L2 Cache Size: " << format_memory_size(properties_.l2CacheSize) << "\n";
    oss << "Constant Memory: " << format_memory_size(properties_.totalConstMem) << "\n";
    oss << "Registers per Block: " << properties_.regsPerBlock;
    
    return oss.str();
}

std::string GPUDeviceProperties::get_multiprocessor_info() const
{
    if (!is_initialized_) return "Unknown";
    
    std::ostringstream oss;
    oss << "Multiprocessor Count: " << properties_.multiProcessorCount << "\n";
    oss << "Max Threads per Block: " << properties_.maxThreadsPerBlock << "\n";
    oss << "Max Threads per SM: " << properties_.maxThreadsPerMultiProcessor << "\n";
    oss << "Max Blocks per SM: " << properties_.maxBlocksPerMultiProcessor << "\n";
    oss << "Warp Size: " << properties_.warpSize << "\n";
    oss << "Max Grid Size: [" << properties_.maxGridSize[0] << ", " 
        << properties_.maxGridSize[1] << ", " << properties_.maxGridSize[2] << "]\n";
    oss << "Max Block Dimensions: [" << properties_.maxThreadsDim[0] << ", " 
        << properties_.maxThreadsDim[1] << ", " << properties_.maxThreadsDim[2] << "]";
    
    return oss.str();
}

std::string GPUDeviceProperties::get_bandwidth_info() const
{
    if (!is_initialized_) return "Unknown";
    
    std::ostringstream oss;
    oss << "Memory Clock Rate: " << properties_.memoryClockRate / 1000 << " MHz\n";
    oss << "Memory Bus Width: " << properties_.memoryBusWidth << " bits\n";
    
    // Calculate theoretical memory bandwidth
    double bandwidth_gbps = (properties_.memoryClockRate * 1e-3) * 
                           (properties_.memoryBusWidth / 8.0) * 2.0 / 1000.0;
    oss << "Theoretical Memory Bandwidth: " << std::fixed << std::setprecision(1) 
        << bandwidth_gbps << " GB/s\n";
    
    oss << "L2 Cache Size: " << format_memory_size(properties_.l2CacheSize) << "\n";
    oss << "Persisting L2 Cache Size: " << format_memory_size(properties_.persistingL2CacheMaxSize);
    
    return oss.str();
}

std::string GPUDeviceProperties::get_architecture_info() const
{
    if (!is_initialized_) return "Unknown";
    
    std::ostringstream oss;
    oss << "Compute Capability: " << properties_.major << "." << properties_.minor << "\n";
    oss << "Architecture: ";
    
    // Map compute capability to architecture name
    if (properties_.major == 8 && properties_.minor == 6) oss << "Ampere (SM86)";
    else if (properties_.major == 8 && properties_.minor == 9) oss << "Ada Lovelace (SM89)";
    else if (properties_.major == 9 && properties_.minor == 0) oss << "Hopper (SM90)";
    else if (properties_.major == 9 && properties_.minor == 0) oss << "Hopper (SM90a)";
    else if (properties_.major == 7 && properties_.minor == 5) oss << "Turing (SM75)";
    else if (properties_.major == 7 && properties_.minor == 0) oss << "Volta (SM70)";
    else if (properties_.major == 6 && properties_.minor == 1) oss << "Pascal (SM61)";
    else if (properties_.major == 5 && properties_.minor == 2) oss << "Maxwell (SM52)";
    else oss << "Unknown Architecture";
    
    oss << "\n";
    oss << "ECC Support: " << (properties_.ECCEnabled ? "Yes" : "No") << "\n";
    oss << "TCC Driver Mode: " << (properties_.tccDriver ? "Yes" : "No") << "\n";
    oss << "Integrated GPU: " << (properties_.integrated ? "Yes" : "No") << "\n";
    oss << "Can Map Host Memory: " << (properties_.canMapHostMemory ? "Yes" : "No");
    
    return oss.str();
}

bool GPUDeviceProperties::supports_unified_memory() const
{
    return is_initialized_ && properties_.unifiedAddressing;
}

bool GPUDeviceProperties::supports_concurrent_kernels() const
{
    return is_initialized_ && properties_.concurrentKernels;
}

bool GPUDeviceProperties::supports_managed_memory() const
{
    return is_initialized_ && properties_.managedMemory;
}

bool GPUDeviceProperties::supports_cooperative_launch() const
{
    return is_initialized_ && properties_.cooperativeLaunch;
}

bool GPUDeviceProperties::supports_multi_device_atomics() const
{
    return is_initialized_ && properties_.multiDeviceAtomics;
}

std::string GPUDeviceProperties::format_bandwidth(double bandwidth_gbps) const
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << bandwidth_gbps << " GB/s";
    return oss.str();
}

} // namespace DeviceManagement
} // namespace Utilities
*/