#include "evaluation.hpp"
#include <filesystem>
#include <numeric>
#include <algorithm>
#include <fstream>

namespace fs = std::filesystem;
using namespace torch::indexing;

void splitCameras(const std::vector<Camera>& all,
                  std::vector<Camera>& train,
                  std::vector<Camera>& test,
                  size_t everyN){
    // Build index vector
    std::vector<size_t> indices(all.size());
    std::iota(indices.begin(), indices.end(), 0);
    // Sort indices by filename
    std::sort(indices.begin(), indices.end(), [&all](size_t a, size_t b){
        return fs::path(all[a].filePath).filename().string() < fs::path(all[b].filePath).filename().string();
    });

    for(size_t i = 0; i < indices.size(); ++i){
        if((i % everyN) == 0) test.push_back(all[indices[i]]);
        else                  train.push_back(all[indices[i]]);
    }
}

static void accumulate(EvalGroup& grp,
                       float psnrV, float ssimV, float l1V, float mLoss){
    grp.meanPSNR += psnrV;
    grp.meanSSIM += ssimV;
    grp.meanL1   += l1V;
    grp.mainLoss += mLoss;
}

static void finalize(EvalGroup& grp, size_t count){
    if(count == 0) return;
    grp.meanPSNR /= count;
    grp.meanSSIM /= count;
    grp.meanL1   /= count;
    grp.mainLoss /= count;
}

EvalSnapshot evaluate(Model& model,
                      std::vector<Camera>& train,
                      std::vector<Camera>& test,
                      size_t iter,
                      size_t maxIter,
                      float ssimWeight,
                      size_t rngSeed){
    // Allow gradients during forward because Model::forward unconditionally
    // retains gradients on some tensors. We will not perform backward here, so
    // memory overhead is minimal.

    EvalSnapshot snap;
    snap.iter = iter;
    snap.percent = 100.0f * static_cast<float>(iter) / static_cast<float>(maxIter);

    // Determine subset of training cameras (10 %)
    size_t subsetSize = std::max<size_t>(1, train.size() / 10);
    std::vector<size_t> trainIdx(train.size());
    std::iota(trainIdx.begin(), trainIdx.end(), 0);
    std::mt19937 rng(rngSeed + iter); // vary with iter so sample can change per milestone
    std::shuffle(trainIdx.begin(), trainIdx.end(), rng);
    trainIdx.resize(subsetSize);

    // Evaluate training subset
    for(size_t idx : trainIdx){
        Camera& cam = train[idx];
        torch::Tensor rgb  = model.forward(cam, static_cast<int>(iter));
        torch::Tensor gt   = cam.getImage(model.getDownscaleFactor(static_cast<int>(iter))).to(model.device);

        float p  = psnr(rgb, gt).item<float>();
        float l1v = l1(rgb, gt).item<float>();
        float ssimV = model.ssim.eval(rgb, gt).item<float>();
        float mLoss = model.mainLoss(rgb, gt, ssimWeight).item<float>();

        accumulate(snap.train, p, ssimV, l1v, mLoss);
    }
    finalize(snap.train, subsetSize);

    // Evaluate all test cameras
    for(Camera& cam : test){
        torch::Tensor rgb  = model.forward(cam, static_cast<int>(iter));
        torch::Tensor gt   = cam.getImage(model.getDownscaleFactor(static_cast<int>(iter))).to(model.device);

        float p  = psnr(rgb, gt).item<float>();
        float l1v = l1(rgb, gt).item<float>();
        float ssimV = model.ssim.eval(rgb, gt).item<float>();
        float mLoss = model.mainLoss(rgb, gt, ssimWeight).item<float>();

        accumulate(snap.test, p, ssimV, l1v, mLoss);
    }
    finalize(snap.test, test.size());

    return snap;
}

void saveEval(const std::vector<EvalSnapshot>& snapshots,
              const std::vector<Camera>& train,
              const std::vector<Camera>& test,
              const std::string& baseScenePath){
    if(snapshots.empty()) return;

    fs::path p(baseScenePath);
    p.replace_filename(p.stem().string() + "_eval.json");

    nlohmann::json jSnap = nlohmann::json::array();

    for(const auto& s : snapshots){
        nlohmann::json jt;
        jt["iter"] = s.iter;
        jt["percent"] = s.percent;

        auto toJson = [](const EvalGroup& g){
            nlohmann::json jg;
            jg["psnr"] = g.meanPSNR;
            jg["ssim"] = g.meanSSIM;
            jg["l1"]   = g.meanL1;
            jg["mainLoss"] = g.mainLoss;
            return jg;
        };

        jt["train"] = toJson(s.train);
        jt["test"]  = toJson(s.test);
        jSnap.push_back(jt);
    }

    // Gather file lists once
    auto camsToFileList = [](const std::vector<Camera>& cams){
        std::vector<std::string> files;
        files.reserve(cams.size());
        for(const auto& c : cams)
            files.push_back(fs::path(c.filePath).filename().string());
        return files;
    };

    nlohmann::json root;
    root["train_files"] = camsToFileList(train);
    root["test_files"]  = camsToFileList(test);
    root["snapshots"]   = jSnap;

    std::ofstream of(p);
    of << root.dump(2);
    of.close();

    std::cout << "Saved evaluation data to " << p << std::endl;
} 