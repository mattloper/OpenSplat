#include "pose_utils.hpp"
#include <iostream> // For std::cout, std::cerr
#include <map>      // For std::map to find common cameras by filename
#include <filesystem> // For fs::path operations

// Anonymous namespace for helper functions local to this file
namespace {
    // Helper to extract camera center (translation) from a 4x4 pose matrix
    torch::Tensor get_camera_center(const torch::Tensor& cam_to_world_matrix) {
        // Assuming cam_to_world_matrix is a 4x4 tensor
        // The translation is in the last column, first 3 rows
        return cam_to_world_matrix.index({torch::indexing::Slice(0, 3), 3});
    }
}

AlignmentResult calculate_alignment_transform(
    const std::vector<Camera>& cameras_a,
    const std::vector<Camera>& cameras_b,
    const torch::Device& operation_device, // Device for operations, can be forced to CPU
    int min_common_cameras) {

    AlignmentResult result;
    // Always perform alignment calculations on CPU to avoid MPS issues with SVD/det
    const torch::Device cpu_device = torch::kCPU;

    result.R = torch::eye(3, torch::TensorOptions().device(cpu_device).dtype(torch::kFloat32));
    result.t = torch::zeros({3}, torch::TensorOptions().device(cpu_device).dtype(torch::kFloat32));
    result.s = 1.0f;
    result.success = false;
    result.common_cameras_count = 0;

    // Find common cameras based on filename
    std::map<std::string, const Camera*> map_cameras_a;
    for (const auto& cam_a : cameras_a) {
        map_cameras_a[fs::path(cam_a.filePath).filename().string()] = &cam_a;
    }

    std::vector<torch::Tensor> points_a_list;
    std::vector<torch::Tensor> points_b_list;

    for (const auto& cam_b : cameras_b) {
        std::string filename_b = fs::path(cam_b.filePath).filename().string();
        auto it_a = map_cameras_a.find(filename_b);
        if (it_a != map_cameras_a.end()) {
            const Camera* cam_a = it_a->second;
            // Move to CPU for alignment processing
            points_a_list.push_back(get_camera_center(cam_a->camToWorld.to(cpu_device)));
            points_b_list.push_back(get_camera_center(cam_b.camToWorld.to(cpu_device)));
        }
    }

    result.common_cameras_count = points_a_list.size();
    if (result.common_cameras_count < min_common_cameras) {
        std::cerr << "Warning: Found only " << result.common_cameras_count
                  << " common cameras. Need at least " << min_common_cameras
                  << " for alignment. Skipping alignment." << std::endl;
        // Result tensors are already on cpu_device, ensure they are moved to original operation_device if needed by caller
        // For now, this function will return CPU tensors for R and t. Main will handle final device placement.
        result.R = result.R.to(operation_device);
        result.t = result.t.to(operation_device);
        return result; 
    }

    std::cout << "Found " << result.common_cameras_count << " common cameras for alignment. Processing on CPU." << std::endl;

    // Stack points into Nx3 tensors
    torch::Tensor P_A = torch::stack(points_a_list, 0); // Already on CPU
    torch::Tensor P_B = torch::stack(points_b_list, 0); // Already on CPU

    // 1. Calculate centroids
    torch::Tensor centroid_A = torch::mean(P_A, 0);
    torch::Tensor centroid_B = torch::mean(P_B, 0);

    // 2. Center points
    torch::Tensor P_A_centered = P_A - centroid_A;
    torch::Tensor P_B_centered = P_B - centroid_B;

    // 3. Compute covariance matrix H = P_B_centered^T * P_A_centered
    torch::Tensor H = torch::matmul(P_B_centered.mT(), P_A_centered);

    // 4. Perform SVD on H: U, S_vec, V = SVD(H). V from torch::linalg_svd is V, not V.T()
    auto svd_result = torch::linalg_svd(H); // SVD on CPU
    torch::Tensor U = std::get<0>(svd_result);
    // torch::Tensor S_vec = std::get<1>(svd_result); // Singular values vector
    torch::Tensor V = std::get<2>(svd_result);

    // 5. Calculate rotation R = V @ U.mT()
    torch::Tensor R_calc = torch::matmul(V, U.mT());

    // Ensure a proper rotation matrix (handle potential reflections)
    if (torch::linalg_det(R_calc).item<float>() < 0) { // linalg_det on CPU
        std::cout << "Reflection detected in rotation matrix. Correcting..." << std::endl;
        torch::Tensor V_copy = V.clone();
        V_copy.index_put_({torch::indexing::Slice(), V_copy.size(1) - 1}, V_copy.index({torch::indexing::Slice(), V_copy.size(1) - 1}) * -1);
        R_calc = torch::matmul(V_copy, U.mT());
    }
    result.R = R_calc; // R is on CPU

    // 6. Calculate scale
    // s = dot(P_A_centered_flat, (P_B_centered @ R.T)_flat) / dot(P_B_centered_flat, P_B_centered_flat)
    torch::Tensor s_tensor = torch::dot(P_A_centered.flatten(), torch::matmul(P_B_centered, result.R.mT()).flatten()) /
                             torch::dot(P_B_centered.flatten(), P_B_centered.flatten());
    result.s = s_tensor.item<float>();

    // 7. Calculate translation t = centroid_A - s * R @ centroid_B
    // centroid_B needs to be [3,1] for matmul with R [3,3]
    result.t = centroid_A - result.s * torch::matmul(result.R, centroid_B.unsqueeze(1)).squeeze(); // t is on CPU

    result.success = true;
    std::cout << "Alignment successful (on CPU). Scale: " << result.s << std::endl;
    // std::cout << "Rotation:\n" << result.R << std::endl;
    // std::cout << "Translation:\n" << result.t << std::endl;

    // Move results to the originally requested device before returning
    result.R = result.R.to(operation_device);
    result.t = result.t.to(operation_device);

    return result;
} 