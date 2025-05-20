#ifndef POSE_UTILS_HPP
#define POSE_UTILS_HPP

#include <vector>
#include <string>
#include <torch/torch.h>
#include "input_data.hpp" // For the Camera struct

namespace fs = std::filesystem; // For fs::path().filename()

// Struct to hold the result of the alignment
struct AlignmentResult {
    torch::Tensor R; // 3x3 rotation matrix
    torch::Tensor t; // 3x1 translation vector
    float s;         // scale factor
    bool success;    // Flag to indicate if alignment was successful (e.g., enough points)
    int common_cameras_count; // Number of common cameras found
};

/**
 * @brief Calculates the similarity transformation (scale, rotation, translation)
 *        to align camera poses from coordinate system B to coordinate system A.
 *
 * This function uses Horn's method (absolute orientation with scale) based on
 * corresponding camera centers (translation components of camToWorld matrices).
 *
 * @param cameras_a A vector of Camera objects from coordinate system A.
 *                  Poses are assumed to be in a normalized, OpenGL-like convention.
 * @param cameras_b A vector of Camera objects from coordinate system B.
 *                  Poses are assumed to be in a normalized, OpenGL-like convention.
 * @param device The torch::Device to perform tensor operations on.
 * @param min_common_cameras The minimum number of common cameras required for a successful alignment.
 * @return AlignmentResult containing the 3x3 rotation matrix R, 3x1 translation vector t,
 *         scale factor s, and a success flag.
 *         If alignment fails (e.g. not enough common cameras), R will be identity, t zero, s one, and success false.
 */
AlignmentResult calculate_alignment_transform(
    const std::vector<Camera>& cameras_a,
    const std::vector<Camera>& cameras_b,
    const torch::Device& device,
    int min_common_cameras = 3
);

#endif // POSE_UTILS_HPP 