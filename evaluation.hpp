#ifndef EVALUATION_H
#define EVALUATION_H

#include <vector>
#include <string>
#include <random>
#include <nlohmann/json.hpp>
#include "input_data.hpp"
#include "model.hpp"

struct EvalGroup{
    float meanPSNR = 0.0f;
    float meanSSIM = 0.0f;
    float meanL1   = 0.0f;
    float mainLoss = 0.0f;
    float meanVarLap = 0.0f; // Variance of Laplacian
    float meanTenengrad = 0.0f; // Tenengrad focus metric
};

struct EvalSnapshot{
    size_t iter;     // absolute iteration number
    float  percent;  // percent progress (0-100)
    EvalGroup train;
    EvalGroup test;
};

// Deterministic alphabetical split: everyNth frame goes to test.
void splitCameras(const std::vector<Camera>& all,
                  std::vector<Camera>& train,
                  std::vector<Camera>& test,
                  size_t everyN = 10);

// Evaluate model on camera sets. Uses 10 % random subset of train for speed.
EvalSnapshot evaluate(Model& model,
                      std::vector<Camera>& train,
                      std::vector<Camera>& test,
                      size_t iter,
                      size_t maxIter,
                      float ssimWeight,
                      size_t rngSeed = 42);

// Persist snapshots to <scene>_eval.json.
void saveEval(const std::vector<EvalSnapshot>& snapshots,
              const std::vector<Camera>& train,
              const std::vector<Camera>& test,
              const std::string& baseScenePath);

#endif 