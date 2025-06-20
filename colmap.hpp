#ifndef COLMAP_H
#define COLMAP_H

#include <fstream>
#include "input_data.hpp"

namespace cm{
    // Back-compat 2-argument version (kept for legacy code)
    InputData inputDataFromColmap(const std::string &projectRoot,
                                  const std::string &colmapImageSourcePath = "");

    // New extended version that allows overriding normalization.
    InputData inputDataFromColmap(const std::string &projectRoot,
                                  const std::string &colmapImageSourcePath,
                                  const std::vector<float> &overrideTranslation,
                                  float overrideScale = -1.0f);

    enum CameraModel{
        SimplePinhole = 0, Pinhole, SimpleRadial, Radial,
        OpenCV, OpenCVFisheye, FullOpenCV, FOV, 
        SimpleRadialFisheye, RadialFisheye, ThinPrismFisheye
    };
}

#endif