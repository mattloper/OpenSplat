#ifndef NANMEAN_HPP
#define NANMEAN_HPP

#include <torch/torch.h>

namespace custom_ops {

// Computes the mean of all non-NaN elements in the input tensor.
// If all elements are NaN, or the tensor is empty, the result is NaN.
// This function reduces over all dimensions of the input tensor.
torch::Tensor nanmean(const torch::Tensor& input);

} // namespace custom_ops

#endif // NANMEAN_HPP 