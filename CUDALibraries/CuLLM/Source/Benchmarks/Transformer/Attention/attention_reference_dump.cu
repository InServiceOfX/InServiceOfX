//------------------------------------------------------------------------------
/// Dump CuLLM FlashAttention output for cross-language comparison.
///
/// This executable is intentionally small and deterministic.  It generates the
/// same Q/K/V values as CUDALibraries/CuLLM/Python/jax_attention_reference.py,
/// runs the warp-cooperative FlashAttention kernel, and writes the output as
/// plain text:
///
///   n head_dim causal
///   o_0
///   o_1
///   ...
///
/// It is used by compare_cullm_jax_attention.py to compare CUDA C++ output
/// directly against JAX on identical inputs.
//------------------------------------------------------------------------------

#include "DataStructures/Array.h"
#include "Transformer/Attention/flash_attention_warp_cooperative.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using DataStructures::Array;
using std::string;
using std::vector;
using Transformer::Attention::flash_attention_warp_cooperative;

namespace
{

vector<float> make_inputs(const int count, const int seed)
{
  vector<float> result(count);
  for (int i {0}; i < count; ++i)
  {
    result[i] =
      static_cast<float>((i * seed + 7 * (i % 13)) % 211 - 105) / 105.0f;
  }
  return result;
}

struct Options
{
  int sequence_length {128};
  int head_dim {64};
  bool causal {false};
  string output_path {"/tmp/cullm_attention_output.txt"};
};

Options parse_options(const int argc, char** argv)
{
  Options options {};
  for (int i {1}; i < argc; ++i)
  {
    const string arg {argv[i]};
    const auto require_value =
      [&](const char* name) -> string
      {
        if (i + 1 >= argc)
        {
          throw std::runtime_error(string("missing value for ") + name);
        }
        return string(argv[++i]);
      };

    if (arg == "--n")
    {
      options.sequence_length = std::atoi(require_value("--n").c_str());
    }
    else if (arg == "--head-dim")
    {
      options.head_dim = std::atoi(require_value("--head-dim").c_str());
    }
    else if (arg == "--causal")
    {
      options.causal = std::atoi(require_value("--causal").c_str()) != 0;
    }
    else if (arg == "--output")
    {
      options.output_path = require_value("--output");
    }
    else
    {
      throw std::runtime_error("unknown option: " + arg);
    }
  }
  return options;
}

template <int kHeadDim, bool kCausal>
void run_case(const Options& options)
{
  const int n {options.sequence_length};
  const vector<float> queries {make_inputs(n * kHeadDim, 3)};
  const vector<float> keys {make_inputs(n * kHeadDim, 5)};
  const vector<float> values {make_inputs(n * kHeadDim, 11)};

  Array<float> d_queries(n * kHeadDim);
  Array<float> d_keys(n * kHeadDim);
  Array<float> d_values(n * kHeadDim);
  Array<float> d_output(n * kHeadDim);
  Array<float> d_logsumexp(n);
  d_queries.copy_host_input_to_device(queries);
  d_keys.copy_host_input_to_device(keys);
  d_values.copy_host_input_to_device(values);

  flash_attention_warp_cooperative<float, kHeadDim, 8, kCausal>(
    d_output.elements_,
    d_logsumexp.elements_,
    d_queries.elements_,
    d_keys.elements_,
    d_values.elements_,
    n);
  cudaDeviceSynchronize();

  vector<float> output(n * kHeadDim);
  d_output.copy_device_output_to_host(output);

  std::ofstream file(options.output_path);
  if (!file)
  {
    throw std::runtime_error("could not open output file: " + options.output_path);
  }
  file.setf(std::ios::scientific);
  file.precision(9);
  file << n << " " << kHeadDim << " " << (kCausal ? 1 : 0) << "\n";
  for (const float value : output)
  {
    file << value << "\n";
  }
}

} // namespace

int main(int argc, char** argv)
{
  try
  {
    const Options options {parse_options(argc, argv)};
    if (options.head_dim == 32 && !options.causal)
    {
      run_case<32, false>(options);
    }
    else if (options.head_dim == 32 && options.causal)
    {
      run_case<32, true>(options);
    }
    else if (options.head_dim == 64 && !options.causal)
    {
      run_case<64, false>(options);
    }
    else if (options.head_dim == 64 && options.causal)
    {
      run_case<64, true>(options);
    }
    else
    {
      throw std::runtime_error("supported head dimensions are 32 and 64");
    }
  }
  catch (const std::exception& error)
  {
    std::cerr << "AttentionReferenceDump error: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
