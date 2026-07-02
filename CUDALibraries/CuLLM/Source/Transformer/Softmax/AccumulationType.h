#ifndef TRANSFORMER_SOFTMAX_ACCUMULATION_TYPE_H
#define TRANSFORMER_SOFTMAX_ACCUMULATION_TYPE_H

namespace Transformer
{
namespace Softmax
{

//------------------------------------------------------------------------------
/// Maps the I/O type T to the accumulation precision for softmax reductions.
///
/// Default (float, __half, bfloat16, ...): accumulate in float.
///   sum = Σ exp(x_i - max_value) is bounded by C (the row length) since every
///   term exp(x_i - max_value) ∈ [0,1]. For any realistic sequence length C,
///   sum << FLT_MAX, so overflow is not a concern.
///
/// double: accumulate in double, preserving the precision T = double was chosen
///   for. Using float here would corrupt max_value and therefore every
///   x_i - max_value subtraction.
//------------------------------------------------------------------------------
template <typename T>
struct AccumulationType
{
  using type = float;
};

template <>
struct AccumulationType<double>
{
  using type = double;
};

template <typename T>
using accumulation_type_t = typename AccumulationType<T>::type;

} // namespace Softmax
} // namespace Transformer

#endif // TRANSFORMER_SOFTMAX_ACCUMULATION_TYPE_H
