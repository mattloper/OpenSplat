#include "nanmean.hpp"
#include <limits> // Required for std::numeric_limits
#include <stdexcept> // Required for std::invalid_argument

namespace custom_ops {

torch::Tensor nanmean(const torch::Tensor& input) {
    if (!input.is_floating_point()) {
        throw std::invalid_argument("custom_ops::nanmean expects a floating-point tensor as input.");
    }

    // Create a mask for non-NaN values (true where not NaN, false where NaN)
    torch::Tensor non_nan_mask = torch::isfinite(input); // isfinite is true for numbers, false for NaN and Inf
                                                   // For strict NaN checking: !torch::isnan(input);
                                                   // isfinite is generally safer if Infs could also appear and should be excluded.

    // Convert boolean mask to the dtype of the input tensor (or a float type for summation)
    torch::Tensor non_nan_mask_float = non_nan_mask.to(input.dtype());

    // Sum of non-NaN elements: multiply input by the 0/1 mask (NaNs become 0)
    // then sum. This avoids NaNs propagating in the sum.
    // An alternative to `where` is `input.clone().masked_fill_(torch::isnan(input), 0.0)`.sum()`
    torch::Tensor sum_of_elements = torch::where(non_nan_mask, input, torch::tensor(0.0, input.options())).sum();

    // Count of non-NaN elements
    torch::Tensor count_of_non_nans = non_nan_mask_float.sum();

    // Avoid division by zero if all elements are NaN or the tensor is empty
    if (count_of_non_nans.item<double>() == 0) {
        return torch::tensor(std::numeric_limits<double>::quiet_NaN(), input.options());
    }

    return sum_of_elements / count_of_non_nans;
}

} // namespace custom_ops 