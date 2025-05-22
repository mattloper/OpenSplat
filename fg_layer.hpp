#pragma once
#include <torch/torch.h>
#include <filesystem>
#include <memory>

struct FgLayer {
    /* Construction */
    FgLayer() = default;
    explicit FgLayer(const torch::Tensor &maskCpu);

    void toDevice(const torch::Device &dev);

    /* Training hooks */
    void zeroGrad();
    void step();
    void setupOptimizer(); // Centralized method to create the optimizer

    /* Forward path */
    torch::Tensor composite(const torch::Tensor &rgbSplat, int downScaleFactor, const torch::Tensor &viewDirWorld) const;

    /* IO */
    void save(const std::string &filename) const;
    bool loadFromText(const std::string &data_txt_filename, const torch::Device &device, bool for_training = false);

    /* Helper */
    bool enabled() const { return enabled_; }
    int height() const { return H_; }
    int width() const { return W_; }

    void initialiseFromGroundTruth(const torch::Tensor &gtFullRes /*H×W×3 float [0,1] on device*/);

private:
    // immutable mask info
    torch::Tensor maskBool_;   // H×W×1 bool on CPU
    torch::Tensor indices_;    // N long (flattened) on device

    // trainable params (on device)
    torch::Tensor opacity_;         // N
    torch::Tensor colorCoeffs_;     // N×4×3 (basis × RGB)

    int H_{0}, W_{0};
    bool enabled_{false};

    std::unique_ptr<torch::optim::Adam> opt_;
}; 