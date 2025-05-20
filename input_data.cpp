#include <filesystem>
#include <nlohmann/json.hpp>
#include "input_data.hpp"
#include "cv_utils.hpp"
#include "utils.hpp"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <limits>

namespace fs = std::filesystem;
using namespace torch::indexing;
using json = nlohmann::json;

namespace ns{ InputData inputDataFromNerfStudio(const std::string &projectRoot); }
namespace cm{ InputData inputDataFromColmap(const std::string &projectRoot, const std::string& imageSourcePath); }
namespace osfm { InputData inputDataFromOpenSfM(const std::string &projectRoot); }
namespace omvg { InputData inputDataFromOpenMVG(const std::string &projectRoot); }

InputData inputDataFromX(const std::string &projectRoot, const std::string& colmapImageSourcePath){
    fs::path root(projectRoot);

    if (fs::exists(root / "transforms.json")){
        return ns::inputDataFromNerfStudio(projectRoot);
    }else if (fs::exists(root / "sparse") || fs::exists(root / "cameras.bin")){
        return cm::inputDataFromColmap(projectRoot, colmapImageSourcePath);
    }else if (fs::exists(root / "reconstruction.json")){
        return osfm::inputDataFromOpenSfM(projectRoot);
    }else if (fs::exists(root / "opensfm" / "reconstruction.json")){
        return osfm::inputDataFromOpenSfM((root / "opensfm").string());
    }else if (fs::exists(root / "sfm_data.json")){
        return omvg::inputDataFromOpenMVG((root).string());
    }
    else{
        throw std::runtime_error("Invalid project folder (must be either a colmap or nerfstudio or openmvg project folder)");
    }
}

torch::Tensor Camera::getIntrinsicsMatrix(){
    return torch::tensor({{fx, 0.0f, cx},
                          {0.0f, fy, cy},
                          {0.0f, 0.0f, 1.0f}}, torch::kFloat32);
}

void Camera::loadImage(float downscaleFactor){
    // Populates image and K, then updates the camera parameters
    // Caution: this function has destructive behaviors
    // and should be called only once
    if (image.numel()) std::runtime_error("loadImage already called");
    std::cout << "Loading " << filePath << std::endl;

    cv::Mat cImg = imreadRGB(filePath);
    
    float rescaleF = 1.0f;
    // If camera intrinsics don't match the image dimensions 
    if (cImg.rows != height || cImg.cols != width){
        rescaleF = static_cast<float>(cImg.rows) / static_cast<float>(height);
    }
    fx *= rescaleF;
    fy *= rescaleF;
    cx *= rescaleF;
    cy *= rescaleF;

    if (downscaleFactor > 1.0f){
        float scaleFactor = 1.0f / downscaleFactor;
        cv::resize(cImg, cImg, cv::Size(), scaleFactor, scaleFactor, cv::INTER_AREA);
        fx *= scaleFactor;
        fy *= scaleFactor;
        cx *= scaleFactor;
        cy *= scaleFactor;
    }

    K = getIntrinsicsMatrix();
    cv::Rect roi;

    if (hasDistortionParameters()){
        // Undistort
        std::vector<float> distCoeffs = undistortionParameters();
        cv::Mat cK = floatNxNtensorToMat(K);
        cv::Mat newK = cv::getOptimalNewCameraMatrix(cK, distCoeffs, cv::Size(cImg.cols, cImg.rows), 0, cv::Size(), &roi);

        cv::Mat undistorted = cv::Mat::zeros(cImg.rows, cImg.cols, cImg.type());
        cv::undistort(cImg, undistorted, cK, distCoeffs, newK);
        
        image = imageToTensor(undistorted);
        K = floatNxNMatToTensor(newK);
    }else{
        roi = cv::Rect(0, 0, cImg.cols, cImg.rows);
        image = imageToTensor(cImg);
    }

    // Crop to ROI
    image = image.index({Slice(roi.y, roi.y + roi.height), Slice(roi.x, roi.x + roi.width), Slice()});

    // Update parameters
    height = image.size(0);
    width = image.size(1);
    fx = K[0][0].item<float>();
    fy = K[1][1].item<float>();
    cx = K[0][2].item<float>();
    cy = K[1][2].item<float>();
}

torch::Tensor Camera::getImage(int downscaleFactor){
    if (downscaleFactor <= 1) return image;
    else{

        // torch::jit::script::Module container = torch::jit::load("gt.pt");
        // return container.attr("val").toTensor();

        if (imagePyramids.find(downscaleFactor) != imagePyramids.end()){
            return imagePyramids[downscaleFactor];
        }

        // Rescale, store and return
        cv::Mat cImg = tensorToImage(image);
        cv::resize(cImg, cImg, cv::Size(cImg.cols / downscaleFactor, cImg.rows / downscaleFactor), 0.0, 0.0, cv::INTER_AREA);
        torch::Tensor t = imageToTensor(cImg);
        imagePyramids[downscaleFactor] = t;
        return t;
    }
}

bool Camera::hasDistortionParameters(){
    return k1 != 0.0f || k2 != 0.0f || k3 != 0.0f || p1 != 0.0f || p2 != 0.0f;
}

std::vector<float> Camera::undistortionParameters(){
    std::vector<float> p = { k1, k2, p1, p2, k3, 0.0f, 0.0f, 0.0f };
    return p;
}

std::tuple<std::vector<Camera>, Camera *> InputData::getCameras(bool validate, const std::string &valImage){
    if (!validate) return std::make_tuple(cameras, nullptr);
    else{
        size_t valIdx = -1;
        std::srand(42);

        if (valImage == "random"){
            valIdx = std::rand() % cameras.size();
        }else{
            for (size_t i = 0; i < cameras.size(); i++){
                if (fs::path(cameras[i].filePath).filename().string() == valImage){
                    valIdx = i;
                    break;
                }
            }
            if (valIdx == -1) throw std::runtime_error(valImage + " not in the list of cameras");
        }

        std::vector<Camera> cams;
        Camera *valCam = nullptr;

        for (size_t i = 0; i < cameras.size(); i++){
            if (i != valIdx) cams.push_back(cameras[i]);
            else valCam = &cameras[i];
        }

        return std::make_tuple(cams, valCam);
    }
}


void InputData::saveCameras(const std::string &filename, bool keepCrs){
    json j = json::array();
    
    for (size_t i = 0; i < cameras.size(); i++){
        Camera &cam = cameras[i];

        json camera = json::object();
        camera["id"] = i;
        camera["img_name"] = fs::path(cam.filePath).filename().string();
        camera["width"] = cam.width;
        camera["height"] = cam.height;
        camera["fx"] = cam.fx;
        camera["fy"] = cam.fy;

        torch::Tensor R = cam.camToWorld.index({Slice(None, 3), Slice(None, 3)});
        torch::Tensor T = cam.camToWorld.index({Slice(None, 3), Slice(3,4)}).squeeze();
        
        // Flip z and y
        R = torch::matmul(R, torch::diag(torch::tensor({1.0f, -1.0f, -1.0f})));

        if (keepCrs) T = (T / scale) + translation;

        std::vector<float> position(3);
        std::vector<std::vector<float>> rotation(3, std::vector<float>(3));
        for (int i = 0; i < 3; i++) {
            position[i] = T[i].item<float>();
            for (int j = 0; j < 3; j++) {
                rotation[i][j] = R[i][j].item<float>();
            }
        }

        camera["position"] = position;
        camera["rotation"] = rotation;
        j.push_back(camera);
    }
    
    std::ofstream of(filename);
    of << j;
    of.close();

    std::cout << "Wrote " << filename << std::endl;
}

void Camera::loadMask(const std::string &mask_source_path,
                      bool is_global_mask_mode,
                      const cv::Mat &global_mask_cv_mat){
    // Clear any existing cache
    maskPyramids.clear();
    maskSourceCV.release();
    has_mask = false;

    cv::Mat mask_cv;
    if (is_global_mask_mode){
        if (!global_mask_cv_mat.empty()){
            mask_cv = global_mask_cv_mat.clone();
        }
    } else {
        if (!mask_source_path.empty()){
            fs::path mask_dir = mask_source_path;
            fs::path image_basename = fs::path(this->filePath).stem();
            const std::vector<std::string> exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"};
            for (const auto &ext : exts){
                fs::path candidate = mask_dir / (image_basename.string() + ext);
                if (fs::exists(candidate) && fs::is_regular_file(candidate)){
                    mask_cv = cv::imread(candidate.string(), cv::IMREAD_GRAYSCALE);
                    if (!mask_cv.empty()) break;
                }
            }
        }
    }

    if (!mask_cv.empty()){
        has_mask = true;
        maskSourceCV = mask_cv; // store full-res copy
    }
}

torch::Tensor Camera::mask(int downscaleFactor, const torch::Device &device){
    // Ensure image pyramid exists to know target dims
    torch::Tensor img = getImage(downscaleFactor);
    const int h = img.size(0);
    const int w = img.size(1);

    if (!has_mask){
        return torch::ones({h, w, 1}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
    }

    if (maskPyramids.find(downscaleFactor) != maskPyramids.end()){
        return maskPyramids[downscaleFactor].to(device);
    }

    // Build resized mask tensor from source
    cv::Mat resized;
    if (maskSourceCV.rows != h || maskSourceCV.cols != w){
        cv::resize(maskSourceCV, resized, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);
    } else {
        resized = maskSourceCV;
    }

    torch::Tensor mask_tensor = torch::from_blob(resized.data, {h, w, 1}, torch::kByte).clone();
    maskPyramids[downscaleFactor] = mask_tensor; // cache (CPU tensor)
    return mask_tensor.to(device);
}

torch::Tensor Camera::maskedImage(int downscaleFactor, const torch::Device &device){
    torch::Tensor img = getImage(downscaleFactor).to(device);
    torch::Tensor m = mask(downscaleFactor, device).to(torch::kBool);
    m = m.expand({img.size(0), img.size(1), 3});
    return img.masked_fill(~m, std::numeric_limits<float>::quiet_NaN());
}