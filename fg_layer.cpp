#include "fg_layer.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream> // Required for std::ofstream
#include <iomanip> // Required for std::scientific
using namespace torch::indexing;

namespace fs = std::filesystem;

FgLayer::FgLayer(const torch::Tensor &maskCpu){
    if (!maskCpu.defined() || maskCpu.numel() == 0){
        enabled_ = false; return;
    }
    maskBool_ = maskCpu.to(torch::kBool).cpu().clone();
    H_ = maskBool_.size(0);
    W_ = maskBool_.size(1);
    torch::Tensor flat = maskBool_.view({-1});
    indices_ = torch::nonzero(flat).squeeze();
    enabled_ = indices_.numel() > 0;

    std::cout << "[FgLayer] Loaded mask " << H_ << "x" << W_
              << "  |  active pixels: " << indices_.numel()
              << " (" << 100.0f * static_cast<float>(indices_.numel()) / (static_cast<float>(H_)*W_) << "%)"
              << std::endl;
}

void FgLayer::setupOptimizer() {
    if (!enabled_) return;
    
    // Release existing optimizer if there is one
    opt_.reset();
    
    float learning_rate = .01;
    
    // This is now the ONLY place where the optimizer is created
    // All other methods must call this method instead of creating their own optimizer
    opt_ = std::make_unique<torch::optim::Adam>(
        std::vector<torch::Tensor>{colorCoeffs_, opacity_}, 
        torch::optim::AdamOptions(learning_rate)
    );
    
    
}

void FgLayer::toDevice(const torch::Device &dev){
    if (!enabled_) return;
    indices_ = indices_.to(dev);
    if (!opacity_.defined()){
        const int64_t N = indices_.size(0);
        // Initialize with logit(0.5) = 0.0 for semi-transparent foreground (50% opacity)
        // Set requires_grad=false to keep opacity fixed during optimization
        opacity_ = torch::full({N}, 0.0f, torch::TensorOptions().device(dev)).requires_grad_(true);
        colorCoeffs_ = torch::zeros({N,4,3}, dev).requires_grad_();

        // Initialize SH1+ with small random noise
        if (colorCoeffs_.size(1) > 1) { // If SH coeffs beyond DC exist (i.e., N,SH,C where SH > 1)
            torch::NoGradGuard nograd_noise; // Don't track gradient for noise generation
            // Create random noise for SH coeffs 1, 2, 3
            torch::Tensor noise = torch::randn({N, 3, 3}, dev) * 0.001f; // N x (SH1,SH2,SH3) x RGB
            if (colorCoeffs_.size(1) > 1) colorCoeffs_.index_put_({Slice(), 1, Slice()}, noise.index({Slice(), 0, Slice()})); // SH1
            if (colorCoeffs_.size(1) > 2) colorCoeffs_.index_put_({Slice(), 2, Slice()}, noise.index({Slice(), 1, Slice()})); // SH2
            if (colorCoeffs_.size(1) > 3) colorCoeffs_.index_put_({Slice(), 3, Slice()}, noise.index({Slice(), 2, Slice()})); // SH3
        }
        
        // Do NOT setup optimizer here - let caller decide when to do that
        std::cout << "[FgLayer] toDevice: Tensors initialized" << std::endl;
    } else {
        opacity_ = opacity_.to(dev);
        colorCoeffs_ = colorCoeffs_.to(dev); // If already defined, assume it's loaded with correct values
        
        // Do NOT setup optimizer here either
        std::cout << "[FgLayer] toDevice: Tensors moved to device" << std::endl;
    }
}

void FgLayer::zeroGrad(){ if (enabled_ && opt_) opt_->zero_grad(); }
void FgLayer::step(){ 
    if (!enabled_ || !opt_) return;
    
    static int step_count = 0;

    // Single optimization step; verbose debugging removed
    opt_->step();
    ++step_count;
}

torch::Tensor FgLayer::composite(const torch::Tensor &rgbSplat, int downScaleFactor, const torch::Tensor &viewDirWorld) const{
    if (!enabled_) return rgbSplat;
    auto device = rgbSplat.device();

    // Compute SH basis (4 components) for this view direction
    torch::Tensor dir = viewDirWorld.to(device);
    dir = dir / dir.norm();
    torch::Tensor basis = torch::empty({4}, device);
    const float c0 = 0.28209479177f;  // 1/2*sqrt(1/pi)
    const float c1 = 0.4886025119f;   // sqrt(3)/(2*sqrt(pi))
    basis[0] = 1.0f; // Use full DC contribution so colour is not scaled by 0.282
    basis[1] = c1 * dir[1]; // Y
    basis[2] = c1 * dir[2]; // Z
    basis[3] = c1 * dir[0]; // X

    torch::Tensor coeffSel = colorCoeffs_.to(device); // N×4×3
    
    // Use all SH components to compute the view-dependent RGB color
    torch::Tensor rgb_fg = torch::zeros({coeffSel.size(0), 3}, device);
    
    // Apply SH basis - loop through all 4 components (DC + 3 SH1 terms)
    for (int64_t i = 0; i < 4; i++) {
        // For each SH component, multiply by the corresponding basis value and add to result
        rgb_fg += coeffSel.index({Slice(), i, Slice()}) * basis[i];
    }
    
    // Apply the standard shift and scale to get final RGB
    rgb_fg = rgb_fg * c0 + 0.5f;

    // Build dense maps
    torch::Tensor opacityFlat = torch::zeros({H_*W_}, device);
    opacityFlat.scatter_(0, indices_.to(device), torch::sigmoid(opacity_));

    torch::Tensor colorFlat = torch::zeros({H_*W_,3}, device);
    torch::Tensor idxExpand = indices_.to(device).view({-1,1}).repeat({1,3});
    colorFlat.scatter_(0, idxExpand, torch::clamp(rgb_fg, 0.0, 1.0));

    torch::Tensor opacityImg = opacityFlat.view({H_,W_,1});
    torch::Tensor colorImg = colorFlat.view({H_,W_,3});

    if (opacityImg.size(0)!=rgbSplat.size(0) || opacityImg.size(1)!=rgbSplat.size(1)){
        // make sure tensors are H×W×C
        if (opacityImg.dim()==2) opacityImg = opacityImg.unsqueeze(-1); // add channel dim
        if (colorImg.dim()==2)   colorImg   = colorImg.unsqueeze(-1).repeat({1,1,3});

        // Resize mask to match rendering resolution (nearest for opacity, bilinear for color)
        opacityImg = torch::nn::functional::interpolate(opacityImg.permute({2,0,1}).unsqueeze(0),
                      torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{rgbSplat.size(0), rgbSplat.size(1)}).mode(torch::kNearest)).squeeze(0).permute({1,2,0});

        colorImg = torch::nn::functional::interpolate(colorImg.permute({2,0,1}).unsqueeze(0),
                      torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{rgbSplat.size(0), rgbSplat.size(1)}).mode(torch::kBilinear).align_corners(false)).squeeze(0).permute({1,2,0});
    }

    if (opacityImg.size(0)==rgbSplat.size(0)&&opacityImg.size(1)==rgbSplat.size(1))
        return opacityImg*colorImg + (1-opacityImg)*rgbSplat;
    else return rgbSplat;
}

void FgLayer::save(const std::string &filename) const{
    if (!enabled_) return;

    // Determine the new text file name, e.g., based on 'filename' but with '.fgdata.txt'
    fs::path base_path(filename);
    // If the input 'filename' was, for example, 'output.fg.pt' or 'output.ply.fg.pt',
    // we want something like 'output.fgdata.txt'.
    // A simple approach is to remove the last extension, then replace the next one,
    // or if it has no extension, append.
    // For now, let's assume filename is a base name like 'output' or 'model_output'
    // and we append '.fgdata.txt' to it if it's not already ending in .pt or .txt
    // A more robust way would be to make 'filename' always be the *base* for outputs.
    // Let's make it so that if filename is 'path/to/output.fg.pt', it becomes 'path/to/output.fgdata.txt'
    
    std::string data_txt_filename;
    if (base_path.has_extension() && (base_path.extension() == ".pt" || base_path.extension() == ".txt")) {
        // Input filename was e.g. "output.fg.pt" or "output.fg.txt"
        // We want "output.fg.txt"
        data_txt_filename = base_path.parent_path() / base_path.stem().stem();
        data_txt_filename += ".fg.txt"; // Ensures .fg.txt extension
    } else { // E.g. filename is "output_iteration_N" (no specific fg extension yet)
        data_txt_filename = base_path.string() + ".fg.txt";
    }


    std::ofstream data_file(data_txt_filename);
    if (!data_file.is_open()){
        std::cerr << "[FgLayer] Failed to open " << data_txt_filename << " for writing." << std::endl;
        return;
    }

    data_file << std::scientific << std::setprecision(8); // Use scientific notation for floats

    // Get CPU copies of tensors
    torch::Tensor opacity_cpu = opacity_.cpu();
    torch::Tensor colorCoeffs_cpu = colorCoeffs_.cpu();
    torch::Tensor indices_cpu = indices_.cpu(); // These are the 1D indices of active points

    int64_t num_points = indices_cpu.size(0);
    if (opacity_cpu.size(0) != num_points || colorCoeffs_cpu.size(0) != num_points) {
        std::cerr << "[FgLayer::save] Error: Mismatch in tensor sizes for saving. "
                  << "Indices: " << num_points
                  << ", Opacity: " << opacity_cpu.size(0)
                  << ", ColorCoeffs: " << colorCoeffs_cpu.size(0) << std::endl;
        data_file.close();
        return;
    }

    // Write header
    data_file << "H " << H_ << "\n";
    data_file << "W " << W_ << "\n";
    data_file << "N_POINTS " << num_points << "\n";
    
    int64_t num_sh_bases = colorCoeffs_cpu.size(1); // Should be 4 (DC, SH1, SH2, SH3)
    int64_t num_channels = colorCoeffs_cpu.size(2); // Should be 3 (RGB)

    data_file << "# Data format: ORIGINAL_FLAT_INDEX OPACITY";
    for (int64_t sh = 0; sh < num_sh_bases; ++sh) {
        for (int64_t ch = 0; ch < num_channels; ++ch) {
            data_file << " SH" << sh << (char)('R'+ch);
        }
    }
    data_file << "\n";

    // Accessors
    auto indices_acc = indices_cpu.accessor<int64_t,1>();
    auto opacity_acc = opacity_cpu.accessor<float,1>();
    auto colorCoeffs_acc = colorCoeffs_cpu.accessor<float,3>(); // N x SH_BASES x CHANNELS

    // Write data per point
    for (int64_t i = 0; i < num_points; ++i) {
        data_file << indices_acc[i]; // Original 1D flat index
        data_file << " " << opacity_acc[i]; // Opacity value (logit space)
        for (int64_t sh = 0; sh < num_sh_bases; ++sh) {
            for (int64_t ch = 0; ch < num_channels; ++ch) {
                data_file << " " << colorCoeffs_acc[i][sh][ch];
            }
        }
        data_file << "\n";
    }

    data_file.close();
    std::cout << "[FgLayer] Saved all foreground layer data to text: " << data_txt_filename << std::endl;
}

// New method to load ALL FgLayer data from the comprehensive text file
bool FgLayer::loadFromText(const std::string &data_txt_filename, const torch::Device &device, bool for_training) {
    if (!fs::exists(data_txt_filename)) {
        std::cerr << "[FgLayer::loadFromText] Error: Data text file does not exist: " << data_txt_filename << std::endl;
        enabled_ = false;
        return false;
    }

    std::ifstream data_file(data_txt_filename);
    if (!data_file.is_open()){
        std::cerr << "[FgLayer::loadFromText] Failed to open " << data_txt_filename << " for reading." << std::endl;
        enabled_ = false;
        return false;
    }

    std::string line, key;
    int parsed_H = -1, parsed_W = -1, parsed_N_POINTS = -1;

    // Parse Header
    while (std::getline(data_file, line)) {
        std::stringstream ss(line);
        ss >> key;
        if (key == "H") {
            ss >> parsed_H;
        } else if (key == "W") {
            ss >> parsed_W;
        } else if (key == "N_POINTS") {
            ss >> parsed_N_POINTS;
        } else if (key == "#" || key.empty()) { // Comment or empty line
            // If it's the data format line, we can break header parsing
            if (line.find("# Data format:") != std::string::npos) {
                break; 
            }
            continue;
        }
        if (parsed_H != -1 && parsed_W != -1 && parsed_N_POINTS != -1) {
            // Break if all headers found, assumes data format line is next or data itself
            if (!std::getline(data_file, line) || line.find("# Data format:") == std::string::npos) {
                 std::cerr << "[FgLayer::loadFromText] Warning: Expected data format line after headers." << std::endl;
                 // It might be missing, so we might be at data already. Reset stream to read this line again if it wasn't a comment.
                 if (data_file.eof() || (line[0] != '#' && !line.empty())) { // if it wasn't a comment, seek back
                    data_file.clear(); // Clear EOF flags
                    data_file.seekg(-(line.length() + 1), std::ios_base::cur); // +1 for newline
                 }
            }
            break;
        }
    }

    if (parsed_H <= 0 || parsed_W <= 0 || parsed_N_POINTS < 0) { // N_POINTS can be 0 for an empty mask
        std::cerr << "[FgLayer::loadFromText] Error: Invalid or missing headers (H, W, N_POINTS) in " << data_txt_filename << std::endl;
        std::cerr << "Parsed H: " << parsed_H << " W: " << parsed_W << " N_POINTS: " << parsed_N_POINTS << std::endl;
        enabled_ = false;
        data_file.close();
        return false;
    }

    H_ = parsed_H;
    W_ = parsed_W;
    int64_t N = parsed_N_POINTS;

    if (N == 0) { // Handle case of empty mask
        std::cout << "[FgLayer::loadFromText] Loaded an empty foreground mask (0 points) from " << data_txt_filename << std::endl;
        indices_ = torch::empty({0}, torch::kInt64).to(device);
        opacity_ = torch::empty({0}, torch::kFloat32).to(device);
        colorCoeffs_ = torch::empty({0, 4, 3}, torch::kFloat32).to(device); // Assuming 4 SH bases, 3 color channels
        enabled_ = true; // Technically enabled, but with no data
        
        // Setup optimizer through the central method if for training
        if (for_training) {
            setupOptimizer();
        }
        
        data_file.close();
        return true;
    }

    std::vector<int64_t> indices_vec;
    std::vector<float> opacity_vec;
    std::vector<float> colorcoeffs_flat_vec;
    int64_t num_sh_bases = 4; // Assuming 4 SH bases (DC, SH1, SH2, SH3)
    int64_t num_channels = 3; // Assuming 3 color channels (RGB)
    int64_t expected_coeffs_per_point = num_sh_bases * num_channels;

    int line_num = 0; // For data lines
    while (std::getline(data_file, line)) {
        if (line.empty() || line[0] == '#') continue; // Skip empty lines or comments
        line_num++;
        std::stringstream ss(line);
        int64_t current_idx;
        float current_opacity;
        std::vector<float> current_coeffs_vec;

        ss >> current_idx;
        ss >> current_opacity;

        float val;
        while(ss >> val) {
            current_coeffs_vec.push_back(val);
        }

        if (current_coeffs_vec.size() != expected_coeffs_per_point) {
            std::cerr << "[FgLayer::loadFromText] Warning: Line " << line_num << " has " << current_coeffs_vec.size()
                      << " color coefficient values, expected " << expected_coeffs_per_point << ". Skipping line." << std::endl;
            continue;
        }

        indices_vec.push_back(current_idx);
        opacity_vec.push_back(current_opacity);
        colorcoeffs_flat_vec.insert(colorcoeffs_flat_vec.end(), current_coeffs_vec.begin(), current_coeffs_vec.end());
    }
    data_file.close();

    if (indices_vec.size() != static_cast<size_t>(N)) {
        std::cerr << "[FgLayer::loadFromText] Error: Number of data lines read (" << indices_vec.size()
                  << ") does not match N_POINTS header (" << N << ") in " << data_txt_filename << std::endl;
        enabled_ = false;
        return false;
    }

    // Create tensors from vectors
    indices_ = torch::from_blob(indices_vec.data(), {N}, torch::kInt64).clone().to(device);
    opacity_ = torch::from_blob(opacity_vec.data(), {N}, torch::kFloat32).clone().to(device);
    colorCoeffs_ = torch::from_blob(colorcoeffs_flat_vec.data(), {N, num_sh_bases, num_channels}, torch::kFloat32).clone().to(device);

    // ALWAYS force opacity to be non-trainable regardless of for_training flag
    opacity_.requires_grad_(true);
    
    if (for_training) {
        colorCoeffs_.requires_grad_(true);
        opacity_.requires_grad_(true);
        
        // Initialize opacity to 50% if requested
        torch::NoGradGuard nograd;
        opacity_.fill_(0.0f);  // logit(0.5) = 0.0, gives 50% opacity
        
        // Setup optimizer through the central method
        setupOptimizer();
    } else {
        opacity_.requires_grad_(true);
        colorCoeffs_.requires_grad_(false);
        
        // STILL force opacity to 50% even in non-training mode
        torch::NoGradGuard nograd;
        opacity_.fill_(0.0f);  // logit(0.5) = 0.0, gives 50% opacity
    }
    
    enabled_ = true;
    std::cout << "[FgLayer] Successfully loaded all foreground data from text: " << data_txt_filename 
              << " (H:" << H_ << ", W:" << W_ << ", Points:" << N << ")" << std::endl;
    return true;
}

void FgLayer::initialiseFromGroundTruth(const torch::Tensor &gtFullRes){
    if (!enabled_) return;
    
    // Ignore ground truth image completely
    auto device = opacity_.device();
    
    torch::NoGradGuard nograd;
    
    // Set opacity to 50% (logit = 0.0)
    opacity_.fill_(0.0f);
    opacity_.requires_grad_(true);
    
    // Fill DC with neutral gray values (0.5)
    colorCoeffs_.index_put_({Slice(), 0, Slice()}, torch::full({colorCoeffs_.size(0), 3}, 0.5f, device));
    
    // Zero out higher order SH coeffs
    if (colorCoeffs_.size(1) > 1) {
        colorCoeffs_.index_put_({Slice(), Slice(1, None), Slice()}, 0.0f);
    }
    
    // Use the central setupOptimizer method instead of creating a new optimizer here
    setupOptimizer();
    
    std::cout << "[FgLayer] initialiseFromGroundTruth(): Using 50% opacity and neutral gray color" << std::endl;
} 