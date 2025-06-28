#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <fstream>
#include <map>
#include <sstream>
#include <cmath>

#include <cxxopts.hpp>
#include "input_data.hpp" // For Camera and InputData if directly used later
#include "utils.hpp"      // For APP_VERSION, utility functions
#include "colmap.hpp"     // For cm::inputDataFromColmap
#include "pose_utils.hpp" // For AlignmentResult and calculate_alignment_transform
#include "cv_utils.hpp" // For tensorToImage
#include "tensor_math.hpp" // For projectionMatrix
#include "spherical_harmonics.hpp" // For SphericalHarmonics(CPU)::apply
#include "project_gaussians.hpp" // For ProjectGaussians(CPU)::apply
#include "rasterize_gaussians.hpp" // For RasterizeGaussians(CPU)::apply
#include "constants.hpp" // For BLOCK_X, BLOCK_Y for TileBounds

// #include "model.hpp" // We will adapt PLY loading, not use Model directly for it.

// Forward declaration for functions that will be in pose_utils.hpp
// struct Transform_B_to_A_type {
//     torch::Tensor R;
//     torch::Tensor t;
//     float s;
// };
// Transform_B_to_A_type calculate_alignment_transform(
//     const std::vector<Camera>& cameras_a,
//     const std::vector<Camera>& cameras_b,
//     const torch::Device& device
// );

// It's good practice for torch includes to be specific if not everything is needed
// or ensure they are consistently included.
#include <torch/torch.h>
#include <opencv2/opencv.hpp> // For cv::Mat, cv::imwrite, cv::cvtColor
#include <atomic>
#include <thread>

namespace fs = std::filesystem;
using namespace torch::indexing;

// Define a simple struct to hold splat data, as planned.
struct SplatData {
    torch::Tensor means;
    torch::Tensor scales;     // log-space
    torch::Tensor quats;
    torch::Tensor featuresDc; // SH DC components
    torch::Tensor featuresRest; // SH Rest components
    torch::Tensor opacities;  // logit-space
    int shDegree = 0; // Will be determined from featuresRest
};

// Local static helper function for projection matrix
static torch::Tensor projectionMatrixRender(float zNear, float zFar, float fovX, float fovY, const torch::Device &device){
    // OpenGL perspective projection matrix (copied from model.cpp's static projectionMatrix)
    float t = zNear * std::tan(0.5f * fovY);
    float b = -t;
    float r = zNear * std::tan(0.5f * fovX);
    float l = -r;
    return torch::tensor({
        {2.0f * zNear / (r - l), 0.0f, (r + l) / (r - l), 0.0f},
        {0.0f, 2.0f * zNear / (t - b), (t + b) / (t - b), 0.0f},
        {0.0f, 0.0f, -(zFar + zNear) / (zFar - zNear), -2.0f * zFar * zNear / (zFar - zNear)},
        {0.0f, 0.0f, -1.0f, 0.0f}
    }, torch::TensorOptions().device(device).dtype(torch::kFloat32));
}

// Static helper function for loading PLY data - to be implemented
static SplatData load_splat_data_from_ply(const std::string& ply_path, torch::Device device) {
    SplatData splat_data;
    std::ifstream f(ply_path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("load_splat_data_from_ply: Cannot open PLY file: " + ply_path);
    }

    std::string line;
    int num_points = 0;
    long long header_bytes = 0;

    // Read header
    std::getline(f, line); header_bytes += line.length() + 1;
    if (line != "ply") throw std::runtime_error("Invalid PLY file: missing \"ply\" keyword.");

    std::getline(f, line); header_bytes += line.length() + 1;
    if (line != "format binary_little_endian 1.0") throw std::runtime_error("Invalid PLY file: expected \"format binary_little_endian 1.0\".");

    // Skip comments, find element vertex line
    while (std::getline(f, line)) {
        header_bytes += line.length() + 1;
        if (line.rfind("element vertex ", 0) == 0) {
            num_points = std::stoi(line.substr(std::string("element vertex ").length()));
            break;
        }
        if (line == "end_header") { // Should not happen before element vertex
            throw std::runtime_error("Invalid PLY file: found end_header before element vertex.");
        }
    }
    if (num_points <= 0) throw std::runtime_error("Invalid PLY file: no points found or invalid point count.");

    std::cout << "Loading " << num_points << " gaussians from PLY file: " << ply_path << std::endl;

    // Property names expected (order matters for simple parsing)
    std::vector<std::string> expected_props_prefixes = {
        "property float x", "property float y", "property float z",
        "property float nx", "property float ny", "property float nz", // Normals (will be read but not stored in SplatData)
        "property float f_dc_" // Start of DC features
    };
    // Properties after f_rest (order matters)
    std::vector<std::string> expected_props_suffix = {
        "property float opacity",
        "property float scale_0", "property float scale_1", "property float scale_2",
        "property float rot_0", "property float rot_1", "property float rot_2", "property float rot_3"
    };

    int features_dc_count = 0;
    int features_rest_count = 0;

    // Read and validate fixed properties before f_dc
    for (const auto& prop_prefix : expected_props_prefixes) {
        std::getline(f, line); header_bytes += line.length() + 1;
        if (prop_prefix == "property float f_dc_") { // Special handling to start counting f_dc
            if (line.rfind(prop_prefix, 0) != 0) {
                 throw std::runtime_error("PLY Parsing Error: Expected property starting with '" + prop_prefix + "' but got '" + line + "'");
            }
            features_dc_count++; // Count the first one
        } else if (line != prop_prefix) {
            throw std::runtime_error("PLY Parsing Error: Expected property '" + prop_prefix + "' but got '" + line + "'");
        }
    }
    
    // Count f_dc properties
    while(std::getline(f, line)) {
        header_bytes += line.length() + 1;
        if (line.rfind("property float f_dc_", 0) == 0) {
            features_dc_count++;
        } else {
            break; // Found a non-f_dc property, this line needs to be processed next
        }
    }
    if (features_dc_count == 0) throw std::runtime_error("PLY Parsing Error: No f_dc properties found.");
     // The line that broke the f_dc loop is the first f_rest (or opacity if shDegree is 0)

    // Count f_rest properties
    // Current 'line' holds the first property after f_dc block
    if (line.rfind("property float f_rest_", 0) == 0) {
        features_rest_count++;
        while(std::getline(f, line)) {
            header_bytes += line.length() + 1;
            if (line.rfind("property float f_rest_", 0) == 0) {
                features_rest_count++;
            } else {
                break; // Found a non-f_rest property, this line needs to be processed next
            }
        }
    }
    // Current 'line' holds the first property after f_rest block (should be opacity)

    // Validate suffix properties
    for (const auto& prop_suffix : expected_props_suffix) {
        if (line != prop_suffix) {
             throw std::runtime_error("PLY Parsing Error: Expected property '" + prop_suffix + "' but got '" + line + "'. Current features_rest_count: " + std::to_string(features_rest_count));
        }
        if (prop_suffix == expected_props_suffix.back()) { // Last expected suffix prop
            break; // Don't read past last expected prop
        }
        std::getline(f, line); header_bytes += line.length() + 1;
    }

    // Find end_header
    while (std::getline(f, line)) {
        header_bytes += line.length() + 1;
        if (line == "end_header") {
            break;
        }
    }
    if (line != "end_header") throw std::runtime_error("PLY Parsing Error: 'end_header' not found or not in expected position.");

    // SH Degree calculation (based on Inria's convention where f_rest are Kx3)
    // Total f_rest_count = K*3, so K = f_rest_count / 3.
    // shDegree = sqrt(K+1)-1 if K = (shDegree+1)^2 - 1 for shDegree > 0
    // For shDegree = 0, K = 0. features_rest_count = 0.
    // For shDegree = 1, K = 3. features_rest_count = 3*3 = 9. (No, this is wrong. featuresRest is (sh_degree+1)^2 - 1 components, each is 3 float = (num_coeffs-1)*3 )
    // The Model code does: featuresRestCpu.reshape({numPoints, 3, featuresRestSize/3}).transpose(2, 1)
    // This implies featuresRestSize is (num_coeffs_per_channel_rest * 3_channels).
    // And num_coeffs_per_channel_rest = (shDegree+1)^2 - 1 (since DC is separate).
    if (features_rest_count > 0 && features_rest_count % 3 != 0) {
        throw std::runtime_error("PLY Parsing Error: features_rest_count must be a multiple of 3. Found: " + std::to_string(features_rest_count));
    }
    int num_sh_coeffs_rest_per_channel = features_rest_count / 3;
    // (shDegree+1)^2 -1 = num_sh_coeffs_rest_per_channel
    // (shDegree+1)^2 = num_sh_coeffs_rest_per_channel + 1
    // shDegree+1 = sqrt(num_sh_coeffs_rest_per_channel + 1)
    // shDegree = sqrt(num_sh_coeffs_rest_per_channel + 1) - 1
    if (num_sh_coeffs_rest_per_channel < 0) { // Should not happen if f_rest_count >=0
         throw std::runtime_error("Error calculating SH degree: negative rest coeffs.");
    }
    double sh_degree_double = std::sqrt(static_cast<double>(num_sh_coeffs_rest_per_channel) + 1.0) - 1.0;
    // Check if sh_degree_double is close to an integer
    if (std::abs(sh_degree_double - std::round(sh_degree_double)) > 1e-5) {
        std::cerr << "Warning: Calculated SH degree is not an integer (" << sh_degree_double << "). This might indicate an issue with PLY f_rest properties. Features_rest_count: " << features_rest_count << std::endl;
        // Defaulting or throwing an error might be better. For now, rounding.
    }
    splat_data.shDegree = static_cast<int>(std::round(sh_degree_double));
    if (splat_data.shDegree < 0) splat_data.shDegree = 0; // Ensure non-negative if rounding leads to it
    
    std::cout << "Detected SH Degree: " << splat_data.shDegree << " (from " << features_rest_count << " f_rest properties)" << std::endl;
    if (features_dc_count != 3 && features_dc_count !=1) { // Original viewer supports 1 or 3 DC features
         std::cerr << "Warning: Expected 1 or 3 f_dc features, found " << features_dc_count << std::endl;
    }
    // If f_dc_count is 1, it is typically grayscale, expand to 3. Model::loadPly seems to expect 3 for featuresDcSize.
    // For simplicity, this loader will expect features_dc_count to be 3. Adapt if PLYs with 1 DC need support.
    if (features_dc_count != 3) {
        throw std::runtime_error("PLY Parsing Error: This loader expects 3 f_dc features (f_dc_0, f_dc_1, f_dc_2). Found: " + std::to_string(features_dc_count));
    }


    // Allocate CPU tensors
    auto tensor_options_float_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor means_cpu = torch::empty({num_points, 3}, tensor_options_float_cpu);
    torch::Tensor scales_cpu = torch::empty({num_points, 3}, tensor_options_float_cpu);
    torch::Tensor quats_cpu = torch::empty({num_points, 4}, tensor_options_float_cpu);
    torch::Tensor features_dc_cpu = torch::empty({num_points, features_dc_count}, tensor_options_float_cpu);
    torch::Tensor features_rest_cpu = torch::empty({num_points, features_rest_count}, tensor_options_float_cpu);
    torch::Tensor opacities_cpu = torch::empty({num_points, 1}, tensor_options_float_cpu);
    
    float normals_buffer[3]; // Buffer to read normals into, then discard

    // Read binary data
    for (int i = 0; i < num_points; ++i) {
        f.read(reinterpret_cast<char*>(means_cpu[i].data_ptr<float>()), sizeof(float) * 3);
        f.read(reinterpret_cast<char*>(normals_buffer), sizeof(float) * 3); // Read and discard normals
        f.read(reinterpret_cast<char*>(features_dc_cpu[i].data_ptr<float>()), sizeof(float) * features_dc_count);
        if (features_rest_count > 0) {
            f.read(reinterpret_cast<char*>(features_rest_cpu[i].data_ptr<float>()), sizeof(float) * features_rest_count);
        }
        f.read(reinterpret_cast<char*>(opacities_cpu[i].data_ptr<float>()), sizeof(float) * 1);
        f.read(reinterpret_cast<char*>(scales_cpu[i].data_ptr<float>()), sizeof(float) * 3);
        f.read(reinterpret_cast<char*>(quats_cpu[i].data_ptr<float>()), sizeof(float) * 4);
    }
    f.close();

    // Populate SplatData struct and move to target device
    splat_data.means = means_cpu.to(device);
    splat_data.scales = scales_cpu.to(device); // These are log_scales as per standard PLY format for gaussians
    splat_data.quats = quats_cpu.to(device);
    splat_data.featuresDc = features_dc_cpu.to(device);
    
    if (features_rest_count > 0) {
        // Reshape featuresRest similar to Model::loadPly:
        // featuresRest = featuresRestCpu.reshape({numPoints, 3, featuresRestSize/3}).transpose(2, 1).to(device).requires_grad_();
        // Here featuresRestSize is features_rest_count, and numPoints is num_points.
        // The original reshape implies (num_points, num_channels (3), num_coeffs_per_channel_rest)
        // Then transpose to (num_points, num_coeffs_per_channel_rest, num_channels (3))
        int num_coeffs_per_channel_rest_calc = features_rest_count / 3;
        splat_data.featuresRest = features_rest_cpu.reshape({num_points, 3, num_coeffs_per_channel_rest_calc}).transpose(1, 2).to(device);
    } else {
        // If shDegree is 0, featuresRest should be an empty tensor with appropriate shape for concatenation later if needed, e.g. (N, 0, 3)
        splat_data.featuresRest = torch::empty({num_points, 0, 3}, tensor_options_float_cpu).to(device);
    }
    splat_data.opacities = opacities_cpu.to(device); // These are logit_opacities

    std::cout << "Successfully loaded " << num_points << " gaussians." << std::endl;
    return splat_data;
}

// Function to render a single image
torch::Tensor render_one_image(
    const SplatData& splat_data, 
    const Camera& render_cam, 
    const torch::Device& device,
    const torch::Tensor& background_color) {

    const float fx = render_cam.fx;
    const float fy = render_cam.fy;
    const float cx = render_cam.cx;
    const float cy = render_cam.cy;
    const int height = render_cam.height;
    const int width = render_cam.width;

    if (height <= 0 || width <= 0) {
        throw std::runtime_error("Render camera height and width must be positive.");
    }

    torch::Tensor cam_to_world = render_cam.camToWorld.to(device);
    torch::Tensor R_render_orig = cam_to_world.index({Slice(None, 3), Slice(None, 3)});
    torch::Tensor T_render = cam_to_world.index({Slice(None, 3), Slice(3,4)});

    // Reinstate the Y/Z flip for R_render, as it got the 3D orientation mostly correct.
    torch::Tensor R_render = torch::matmul(R_render_orig, torch::diag(torch::tensor({1.0f, -1.0f, -1.0f}, torch::TensorOptions().device(device).dtype(torch::kFloat32))));

    torch::Tensor R_inv_render = R_render.transpose(0, 1);
    torch::Tensor T_inv_render = torch::matmul(-R_inv_render, T_render);

    torch::Tensor view_mat = torch::eye(4, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    view_mat.index_put_({Slice(None, 3), Slice(None, 3)}, R_inv_render);
    view_mat.index_put_({Slice(None, 3), Slice(3, 4)}, T_inv_render);
        
    float fovX = 2.0f * std::atan(static_cast<float>(width) / (2.0f * fx));
    float fovY = 2.0f * std::atan(static_cast<float>(height) / (2.0f * fy));

    torch::Tensor proj_mat = projectionMatrixRender(0.01f, 1000.0f, fovX, fovY, device);
    
    torch::Tensor sh_colors = torch::cat({
        splat_data.featuresDc.unsqueeze(1),
        splat_data.featuresRest
    }, 1);

    torch::Tensor xys, radii, conics, depths, num_tiles_hit, cov2d, cam_depths, rgb_output;

    if (device == torch::kCPU) {
        auto p = ProjectGaussiansCPU::apply(
            splat_data.means, 
            torch::exp(splat_data.scales), 
            1.0f, 
            splat_data.quats / splat_data.quats.norm(2, {-1}, true), 
            view_mat, 
            torch::matmul(proj_mat, view_mat), 
            fx, fy, cx, cy, 
            height, width);
        xys = p[0];
        radii = p[1];
        conics = p[2];
        cov2d = p[3];
        cam_depths = p[4];
    } else {
        #if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)
        TileBounds tile_bounds = std::make_tuple(
            (width + BLOCK_X - 1) / BLOCK_X,
            (height + BLOCK_Y - 1) / BLOCK_Y,
            1);
        auto p = ProjectGaussians::apply(
            splat_data.means, 
            torch::exp(splat_data.scales),
            1.0f, 
            splat_data.quats / splat_data.quats.norm(2, {-1}, true),
            view_mat, 
            torch::matmul(proj_mat, view_mat),
            fx, fy, cx, cy, 
            height, width, 
            tile_bounds);
        xys = p[0];
        depths = p[1];
        radii = p[2];
        conics = p[3];
        num_tiles_hit = p[4];
        #else
            throw std::runtime_error("render_one_image: GPU support not built, but device is not CPU.");
        #endif
    }
    
    if ((radii.defined() && radii.numel() > 0 && radii.sum().item<float>() == 0.0f && xys.numel() == 0) || 
        (!xys.defined() || xys.numel() == 0)) { 
        return background_color.unsqueeze(0).unsqueeze(0).repeat({height, width, 1});
    }

    torch::Tensor view_dirs = splat_data.means.detach() - cam_to_world.index({Slice(None, 3), 3}).unsqueeze(0).to(device); 
    view_dirs = view_dirs / view_dirs.norm(2, {-1}, true);
    
    torch::Tensor rgbs;
    if (device == torch::kCPU){
        rgbs = SphericalHarmonicsCPU::apply(splat_data.shDegree, view_dirs, sh_colors);
    } else {
        #if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)
        rgbs = SphericalHarmonics::apply(splat_data.shDegree, view_dirs, sh_colors);
        #else
         throw std::runtime_error("render_one_image: GPU support not built for SH, but device is not CPU.");
        #endif
    }
    
    // Remove c0 scaling for background splats to match visualizer behavior
    rgbs = torch::clamp_min(rgbs + 0.5f, 0.0f);

    if (device == torch::kCPU){
        rgb_output = RasterizeGaussiansCPU::apply(
            xys, radii, conics, rgbs, torch::sigmoid(splat_data.opacities),
            cov2d, cam_depths, height, width, background_color);
    } else {  
        #if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)
        rgb_output = RasterizeGaussians::apply(
            xys, depths, radii, conics, num_tiles_hit, 
            rgbs, torch::sigmoid(splat_data.opacities), 
            height, width, background_color);
        #else
         throw std::runtime_error("render_one_image: GPU support not built for Rasterize, but device is not CPU.");
        #endif
    }

    rgb_output = torch::clamp_max(rgb_output, 1.0f);
    return rgb_output;
}

int main(int argc, char *argv[]) {
    // Similar to opensplat.cpp
    // if (argc > 0 && fs::path(argv[0]).filename() == "opensplat-render") { // check needed for CLion/MSVC which passes exe name as first arg
    //      argc--;
    //      argv++;
    // }
    
    cxxopts::Options options("opensplat-render", "OpenSplat Renderer - Renders a .splat file using specified camera views. - " APP_VERSION);
    options.add_options()
        ("splat-file", "Path to the .ply or .splat Gaussian Splatting model file", cxxopts::value<std::string>())
        ("splat-colmap", "Path to the COLMAP directory corresponding to the splat file\'s coordinate system (colmap_A)", cxxopts::value<std::string>())
        ("view-colmap", "Path to the COLMAP directory providing camera viewpoints for rendering (colmap_B). If omitted, splat-colmap is used.", cxxopts::value<std::string>()->default_value(""))
        ("o,output-dir", "Path to the directory where rendered images will be saved", cxxopts::value<std::string>())
        ("device", "Computation device (cpu, cuda, mps). Defaults to best available.", cxxopts::value<std::string>()->default_value(""))
        ("threads", "Number of CPU threads to use for rendering", cxxopts::value<int>()->default_value("8"))
        ("h,help", "Print usage");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "Error parsing options: " << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    // Validate required arguments
    if (!result.count("splat-file") || !result.count("splat-colmap") || !result.count("output-dir")) {
        std::cerr << "Error: Missing required arguments. Use --help for usage information." << std::endl;
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    const std::string splat_file_path = result["splat-file"].as<std::string>();
    const std::string splat_colmap_path = result["splat-colmap"].as<std::string>();
    std::string view_colmap_path = result["view-colmap"].as<std::string>();
    const std::string output_dir_path = result["output-dir"].as<std::string>();
    const std::string fgmask_cli_path = "";
    std::string device_str = result["device"].as<std::string>();
    int num_threads = result["threads"].as<int>();
    if (num_threads <= 0) {
        std::cerr << "Warning: --threads must be positive. Defaulting to 8." << std::endl;
        num_threads = 8;
    }

    if (view_colmap_path.empty()) {
        view_colmap_path = splat_colmap_path;
        std::cout << "INFO: --view-colmap not provided, using --splat-colmap for rendering views: " << view_colmap_path << std::endl;
    }

    // Determine device
    torch::Device device = torch::kCPU;
    if (!device_str.empty()) {
        if (device_str == "cuda" && torch::hasCUDA()) {
            device = torch::kCUDA;
        } else if (device_str == "mps" && torch::hasMPS()) {
            device = torch::kMPS;
        } else if (device_str != "cpu") {
            std::cout << "Warning: Specified device \"" << device_str << "\" not available or recognized. Defaulting to CPU." << std::endl;
        }
    } else { // Auto-detect best available
        if (torch::hasCUDA()) {
            device = torch::kCUDA;
        } else if (torch::hasMPS()) {
            device = torch::kMPS;
        }
    }
    std::cout << "Using device: " << device << std::endl;
    std::cout << "Threads:          " << num_threads << std::endl;


    // Create output directory if it doesn't exist
    if (!fs::exists(output_dir_path)) {
        try {
            if (fs::create_directories(output_dir_path)) {
                std::cout << "Created output directory: " << output_dir_path << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error creating output directory " << output_dir_path << ": " << e.what() << std::endl;
            return EXIT_FAILURE;
        }
    } else if (!fs::is_directory(output_dir_path)) {
         std::cerr << "Error: Output path " << output_dir_path << " exists but is not a directory." << std::endl;
         return EXIT_FAILURE;
    }


    std::cout << "--- Configuration ---\\" << std::endl;
    std::cout << "Splat File:       " << splat_file_path << std::endl;
    std::cout << "Splat COLMAP (A): " << splat_colmap_path << std::endl;
    std::cout << "View COLMAP (B):  " << view_colmap_path << std::endl;
    std::cout << "Output Directory: " << output_dir_path << std::endl;
    std::cout << "Device:           " << device << std::endl;
    std::cout << "---------------------\\" << std::endl;

    try {
        std::cout << "\nStep 1: Loading splat model..." << std::endl;
        SplatData splat_data = load_splat_data_from_ply(splat_file_path, device);
        // Note: load_splat_data_from_ply will throw if the PLY is not as expected.

        std::cout << "\nStep 2: Loading Splat COLMAP data (colmap_A)..." << std::endl;
        InputData input_data_a = cm::inputDataFromColmap(splat_colmap_path, "");
        const std::vector<Camera>& cameras_a = input_data_a.cameras;
        std::cout << "Loaded " << cameras_a.size() << " cameras from " << splat_colmap_path << std::endl;
        std::cout << "Normalization (A) translation=" << input_data_a.translation << ", scale=" << input_data_a.scale << std::endl;
        if (cameras_a.empty()) {
            throw std::runtime_error("No cameras loaded from splat_colmap_path: " + splat_colmap_path);
        }

        std::cout << "\nStep 3: Loading View COLMAP data (colmap_B)..." << std::endl;
        InputData input_data_b; // Declare outside if-else for consistent scope if used later beyond cameras_b
        const std::vector<Camera>* cameras_b_ptr = nullptr;
        if (view_colmap_path == splat_colmap_path) {
            cameras_b_ptr = &cameras_a;
            std::cout << "Using cameras from colmap_A for colmap_B (paths are identical)." << std::endl;
        } else {
            input_data_b = cm::inputDataFromColmap(view_colmap_path, "");
            cameras_b_ptr = &input_data_b.cameras;
            std::cout << "Loaded " << cameras_b_ptr->size() << " cameras from " << view_colmap_path << std::endl;
            std::cout << "Normalization (B) translation=" << input_data_b.translation << ", scale=" << input_data_b.scale << std::endl;
        }
        const std::vector<Camera>& cameras_b = *cameras_b_ptr;
        if (cameras_b.empty()) {
            throw std::runtime_error("No cameras loaded from view_colmap_path: " + view_colmap_path);
        }

        std::cout << "\\nStep 4: Aligning coordinate systems (if needed)...\\" << std::endl;
        torch::Tensor transform_b_to_a_matrix = torch::eye(4, torch::TensorOptions().device(device).dtype(torch::kFloat32)); 
        
        // Always calculate transform for testing purposes, even if paths are the same.
        // if (view_colmap_path != splat_colmap_path) { 
        AlignmentResult align_result = calculate_alignment_transform(cameras_a, cameras_b, device);
        if (align_result.success) {
            transform_b_to_a_matrix.index_put_({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}, align_result.s * align_result.R);
            transform_b_to_a_matrix.index_put_({torch::indexing::Slice(0, 3), 3}, align_result.t);
            std::cout << "Computed alignment transform from colmap_B to colmap_A." << std::endl;
            std::cout << "Alignment common cameras: " << align_result.common_cameras_count << std::endl;
            std::cout << "Alignment Scale: " << align_result.s << std::endl;
        } else {
            std::cerr << "Warning: Alignment calculation reported failure. Using identity transform. Renderings might be misaligned if coordinate systems differ." << std::endl;
            // transform_b_to_a_matrix remains identity if align_result.success is false
        }
        // } else {
        //     std::cout << "colmap_A and colmap_B are the same, no alignment transform needed.\\" << std::endl;
        // }
        std::cout << "\\nFinal Transform_B_to_A matrix being used:\n" << transform_b_to_a_matrix << std::endl;

        std::cout << "\\nStep 5: Rendering...\\" << std::endl;
        torch::Tensor background_color = torch::tensor({0.0f, 0.0f, 0.0f}, torch::TensorOptions().device(device).dtype(torch::kFloat32));

        std::atomic<int> rendered_count{0};
        std::mutex io_mutex;
        const size_t total_cams = cameras_b.size();

        // ------------------------------------------------------------------
        // Multithreading strategy
        // ------------------------------------------------------------------
        // We spawn 'num_threads' worker threads.  An atomic counter 'next_idx'
        // hands out the next camera index to render.  Each worker loops until
        // the counter exceeds 'total_cams', then exits.  This tiny work-stealing
        // pool keeps the code self-contained (no OpenMP / external pools) yet
        // fully utilises the CPU cores.
        //
        // Only stdout and disk I/O (cv::imwrite) touch shared state; they are
        // protected with the single mutex 'io_mutex'.  All tensors and camera
        // structs are read-only inside the workers so they need no locking.
        // ------------------------------------------------------------------

        auto render_one_index = [&](size_t idx) {
            const Camera &cam_b_original = cameras_b[idx];

            torch::Tensor original_cam_b_pose = cam_b_original.camToWorld.to(device);
            torch::Tensor final_render_pose = (splat_colmap_path == view_colmap_path)
                                                ? original_cam_b_pose
                                                : torch::matmul(transform_b_to_a_matrix, original_cam_b_pose);

            Camera render_cam = cam_b_original;
            render_cam.camToWorld = final_render_pose;

            std::string image_filename = fs::path(render_cam.filePath).filename().string();
            {
                std::lock_guard<std::mutex> lock(io_mutex);
                std::cout << "Rendering image for: " << image_filename << " (H:" << render_cam.height << ", W:" << render_cam.width << ")" << std::endl;
            }

            torch::Tensor rendered_image_tensor = render_one_image(splat_data, render_cam, device, background_color);

            // Flip vertically and horizontally to match original orientation
            rendered_image_tensor = torch::flip(rendered_image_tensor, {0, 1});

            cv::Mat image_to_save = tensorToImage(rendered_image_tensor.contiguous().cpu());

            if (image_to_save.channels() == 3) {
                cv::cvtColor(image_to_save, image_to_save, cv::COLOR_RGB2BGR);
            }

            fs::path output_image_path = fs::path(output_dir_path) / (fs::path(image_filename).stem().string() + ".png");
            bool save_ok = false;
            {
                std::lock_guard<std::mutex> lock(io_mutex);
                save_ok = cv::imwrite(output_image_path.string(), image_to_save);
                if (!save_ok) {
                    std::cerr << "Failed to save rendered image to: " << output_image_path << std::endl;
                }
            }
            if (save_ok) {
                rendered_count.fetch_add(1, std::memory_order_relaxed);
            }
        };

        // Atomic counter that gives each worker its next job.
        std::atomic<size_t> next_idx{0};
        auto worker = [&]() {
            while (true) {
                // fetch_add returns the previous value; stop when we run out
                size_t idx = next_idx.fetch_add(1, std::memory_order_relaxed);
                if (idx >= total_cams) break;
                render_one_index(idx);
            }
        };

        std::vector<std::thread> pool;

        // --------------------------------------------------------------
        // Spawn the worker threads.  Each iteration of this loop starts
        // a new OS thread that begins executing the 'worker' lambda.
        // --------------------------------------------------------------
        for (int t = 0; t < num_threads; ++t) {
            pool.emplace_back(worker); // <-- Thread is created here
        }
        for (auto &th : pool) {
            th.join();
        }

        std::cout << "\nSuccessfully rendered and saved " << rendered_count.load() << " images out of " << cameras_b.size() << "." << std::endl;
        std::cout << "opensplat-render finished successfully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Runtime Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
} 