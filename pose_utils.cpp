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
    // Compute on CPU; tensors default to CPU unless specified otherwise
    result.R = torch::eye(3, torch::kFloat32);
    result.t = torch::zeros({3}, torch::kFloat32);
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
            points_a_list.push_back(get_camera_center(cam_a->camToWorld));
            points_b_list.push_back(get_camera_center(cam_b.camToWorld));
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
    torch::Tensor P_A = torch::stack(points_a_list, 0);
    torch::Tensor P_B = torch::stack(points_b_list, 0);

    // 1. Calculate centroids
    torch::Tensor centroid_A = torch::mean(P_A, 0);
    torch::Tensor centroid_B = torch::mean(P_B, 0);

    std::cout << "Centroid A: " << centroid_A << std::endl;
    std::cout << "Centroid B: " << centroid_B << std::endl;
    std::cout << "Centroid distance |A-B|: " << torch::linalg_vector_norm(centroid_A - centroid_B).item<float>() << std::endl;

    // 2. Center points
    torch::Tensor P_A_centered = P_A - centroid_A;
    torch::Tensor P_B_centered = P_B - centroid_B;

    // Print variance (mean squared distance from centroid) for sanity
    float varA = (P_A_centered.pow(2).sum(1)).mean().item<float>();
    float varB = (P_B_centered.pow(2).sum(1)).mean().item<float>();
    std::cout << "Variance of centered sets  A: " << varA << "   B: " << varB << std::endl;

    // 3. Compute covariance matrix H = P_B_centered^T * P_A_centered
    torch::Tensor H = torch::matmul(P_B_centered.mT(), P_A_centered);
    std::cout << "Covariance matrix H:\n" << H << std::endl;

    // 4. Perform SVD on H: U, S_vec, V = SVD(H). V from torch::linalg_svd is V, not V.T()
    auto svd_result = torch::linalg_svd(H); // SVD on CPU
    torch::Tensor U = std::get<0>(svd_result);
    torch::Tensor S_vec = std::get<1>(svd_result); // Singular values vector
    torch::Tensor Vh = std::get<2>(svd_result); // V^T
    torch::Tensor V = Vh.mT(); // Convert to V

    std::cout << "Singular values: " << S_vec << std::endl;

    // 5. Calculate rotation R = V @ U^T
    torch::Tensor R_calc = torch::matmul(V, U.mT());

    // Ensure a proper rotation matrix (handle potential reflections)
    float detR = torch::linalg_det(R_calc).item<float>();
    std::cout << "det(R) before reflection fix: " << detR << std::endl;
    if (detR < 0) {
        std::cout << "Reflection detected in rotation matrix. Correcting..." << std::endl;
        torch::Tensor V_copy = V.clone();
        V_copy.index_put_({torch::indexing::Slice(), V_copy.size(1) - 1}, V_copy.index({torch::indexing::Slice(), V_copy.size(1) - 1}) * -1);
        R_calc = torch::matmul(V_copy, U.mT());
    }
    result.R = R_calc; // R is on CPU

    // 6. Calculate scale using Umeyama formula:  s = sum(S) / sum(||P_B_centered||^2)
    torch::Tensor sum_singular = S_vec.sum();
    torch::Tensor variance_src = P_B_centered.pow(2).sum();
    torch::Tensor s_tensor = sum_singular / variance_src;
    result.s = s_tensor.item<float>();

    std::cout << "Scale numerator (sum singular values): " << sum_singular.item<float>() << std::endl;
    std::cout << "Scale denominator (variance src): " << variance_src.item<float>() << std::endl;

    // 7. Calculate translation t = centroid_A - s * R @ centroid_B
    // centroid_B needs to be [3,1] for matmul with R [3,3]
    result.t = centroid_A - result.s * torch::matmul(result.R, centroid_B.unsqueeze(1)).squeeze(); // t is on CPU

    std::cout << "Debug: First up to 5 paired camera centers (A vs B):" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(5, points_a_list.size()); ++i) {
        std::cout << "  A[" << i << "]: " << points_a_list[i] << "  B[" << i << "]: " << points_b_list[i] << std::endl;
    }

    result.success = true;
    std::cout << "Alignment successful (on CPU). Scale: " << result.s << std::endl;
    std::cout << "Rotation R:\n" << result.R << std::endl;
    std::cout << "Translation t:\n" << result.t << std::endl;

    // Compute RMS alignment error for diagnostic
    torch::Tensor P_B_aligned = result.s * torch::matmul(P_B_centered, result.R.mT());
    torch::Tensor diff = P_A_centered - P_B_aligned;
    float rms_err = torch::sqrt((diff.pow(2).sum(1))).mean().item<float>();
    std::cout << "RMS alignment error: " << rms_err << std::endl;

    // After rms error print, also print first 3 aligned vs target centered points
    for (int i = 0; i < std::min<int>(3, P_A_centered.size(0)); ++i) {
        std::cout << "Sample check i=" << i << "  A_centered: " << P_A_centered[i] << "  aligned B_centered: " << P_B_aligned[i] << std::endl;
    }

    // Move results to the originally requested device before returning
    result.R = result.R.to(operation_device);
    result.t = result.t.to(operation_device);

    return result;
} 