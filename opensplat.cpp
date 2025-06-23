#include <filesystem>
#include <nlohmann/json.hpp>
#include "opensplat.hpp"
#include "input_data.hpp"
#include "utils.hpp"
#include "cv_utils.hpp"
#include "constants.hpp"
#include <cxxopts.hpp>
#include "evaluation.hpp"

#ifdef USE_VISUALIZATION
#include "visualizer.hpp"
#endif

namespace fs = std::filesystem;
using namespace torch::indexing;

int main(int argc, char *argv[]){
    cxxopts::Options options("opensplat", "Open Source 3D Gaussian Splats generator - " APP_VERSION);
    options.add_options()
        ("i,input", "Path to nerfstudio project", cxxopts::value<std::string>())
        ("o,output", "Path where to save output scene", cxxopts::value<std::string>()->default_value("splat.ply"))
        ("s,save-every", "Save output scene every these many steps (set to -1 to disable)", cxxopts::value<int>()->default_value("-1"))
        ("resume", "Resume training from this PLY file", cxxopts::value<std::string>()->default_value(""))
        ("val", "Withhold a camera shot for validating the scene loss")
        ("val-image", "Filename of the image to withhold for validating scene loss", cxxopts::value<std::string>()->default_value("random"))
        ("val-render", "Path of the directory where to render validation images", cxxopts::value<std::string>()->default_value(""))
        ("keep-crs", "Retain the project input's coordinate reference system")
        ("cpu", "Force CPU execution")
        
        ("n,num-iters", "Number of iterations to run", cxxopts::value<int>()->default_value("30000"))
        ("d,downscale-factor", "Scale input images by this factor.", cxxopts::value<float>()->default_value("1"))
        ("num-downscales", "Number of images downscales to use. After being scaled by [downscale-factor], images are initially scaled by a further (2^[num-downscales]) and the scale is increased every [resolution-schedule]", cxxopts::value<int>()->default_value("2"))
        ("resolution-schedule", "Double the image resolution every these many steps", cxxopts::value<int>()->default_value("3000"))
        ("sh-degree", "Maximum spherical harmonics degree (0-4)", cxxopts::value<int>()->default_value("3"))
        ("sh-degree-interval", "Increase the number of spherical harmonics degree after these many steps (will not exceed [sh-degree])", cxxopts::value<int>()->default_value("1000"))
        ("ssim-weight", "Weight to apply to the structural similarity loss. Set to zero to use least absolute deviation (L1) loss only", cxxopts::value<float>()->default_value("0.2"))
        ("ignore-saturated-after", "Iteration after which saturated pixels are ignored (-1 = never)", cxxopts::value<int>()->default_value("-1"))
        ("saturation-threshold", "Pixel value regarded as fully saturated", cxxopts::value<float>()->default_value("1.0"))
        ("refine-every", "Split/duplicate/prune gaussians every these many steps", cxxopts::value<int>()->default_value("100"))
        ("warmup-length", "Split/duplicate/prune gaussians only after these many steps", cxxopts::value<int>()->default_value("500"))
        ("reset-alpha-every", "Reset the opacity values of gaussians after these many refinements (not steps)", cxxopts::value<int>()->default_value("30"))
        ("densify-grad-thresh", "Threshold of the positional gradient norm (magnitude of the loss function) which when exceeded leads to a gaussian split/duplication", cxxopts::value<float>()->default_value("0.0002"))
        ("densify-size-thresh", "Gaussians' scales below this threshold are duplicated, otherwise split", cxxopts::value<float>()->default_value("0.01"))
        ("stop-screen-size-at", "Stop splitting gaussians that are larger than [split-screen-size] after these many steps", cxxopts::value<int>()->default_value("4000"))
        ("split-screen-size", "Split gaussians that are larger than this percentage of screen space", cxxopts::value<float>()->default_value("0.05"))
        ("colmap-image-path", "Override the default image path for COLMAP-based input", cxxopts::value<std::string>()->default_value(""))
        ("colorcal", "Enable per-camera colour calibration")
        ("norm-translation", "Override normalization translation as x,y,z (comma-separated)", cxxopts::value<std::string>()->default_value(""))
        ("norm-scale", "Override normalization scale factor (positive float)", cxxopts::value<float>()->default_value("-1"))

        ("h,help", "Print usage")
        ("version", "Print version")
        ;
    options.parse_positional({ "input" });
    options.positional_help("[colmap/nerfstudio/opensfm/odm/openmvg project path]");
    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("version")){
        std::cout << APP_VERSION << std::endl;
        return EXIT_SUCCESS;
    }
    if (result.count("help") || !result.count("input")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }


    const std::string projectRoot = result["input"].as<std::string>();
    const std::string outputScene = result["output"].as<std::string>();
    const int saveEvery = result["save-every"].as<int>(); 
    const std::string resume = result["resume"].as<std::string>();
    const bool validate = result.count("val") > 0 || result.count("val-render") > 0;
    const std::string valImage = result["val-image"].as<std::string>();
    const std::string valRender = result["val-render"].as<std::string>();
    if (!valRender.empty() && !fs::exists(valRender)) fs::create_directories(valRender);
    const bool keepCrs = result.count("keep-crs") > 0;
    const float downScaleFactor = (std::max)(result["downscale-factor"].as<float>(), 1.0f);
    const int numIters = result["num-iters"].as<int>();
    const int numDownscales = result["num-downscales"].as<int>();
    const int resolutionSchedule = result["resolution-schedule"].as<int>();
    const int shDegree = result["sh-degree"].as<int>();
    // New validation: limit supported SH degree to 4
    if (shDegree > 4) {
        std::cerr << "Error: --sh-degree " << shDegree << " is not supported (maximum is 4)." << std::endl;
        std::cerr << "Higher spherical-harmonic orders are not implemented in the current build." << std::endl;
        return EXIT_FAILURE;
    }
    const int shDegreeInterval = result["sh-degree-interval"].as<int>();
    const float ssimWeight = result["ssim-weight"].as<float>();
    const int ignoreSaturatedAfter = result["ignore-saturated-after"].as<int>();
    const float saturationThreshold = result["saturation-threshold"].as<float>();
    const int refineEvery = result["refine-every"].as<int>();
    const int warmupLength = result["warmup-length"].as<int>();
    const int resetAlphaEvery = result["reset-alpha-every"].as<int>();
    const float densifyGradThresh = result["densify-grad-thresh"].as<float>();
    const float densifySizeThresh = result["densify-size-thresh"].as<float>();
    const int stopScreenSizeAt = result["stop-screen-size-at"].as<int>();
    const float splitScreenSize = result["split-screen-size"].as<float>();
    const std::string colmapImageSourcePath = result["colmap-image-path"].as<std::string>();
    const bool colorcal = result.count("colorcal") > 0;

    // Parse normalization override
    std::vector<float> normTranslation;
    float normScale = result["norm-scale"].as<float>();
    std::string normTrStr = result["norm-translation"].as<std::string>();
    if (!normTrStr.empty()){
        std::stringstream ss(normTrStr);
        std::string item; float val;
        while (std::getline(ss, item, ',')){
            try{ val = std::stof(item); } catch(const std::exception &){ val = NAN; }
            normTranslation.push_back(val);
        }
        if (normTranslation.size() != 3) {
            std::cerr << "Error: --norm-translation must have exactly three comma-separated floats (x,y,z)" << std::endl;
            return EXIT_FAILURE;
        }
        if (normScale <= 0.0f){
            std::cerr << "Error: --norm-scale must be > 0 when --norm-translation is provided" << std::endl;
            return EXIT_FAILURE;
        }
    }

    torch::Device device = torch::kCPU;
    int displayStep = 10;

    if (torch::hasCUDA() && result.count("cpu") == 0) {
        std::cout << "Using CUDA" << std::endl;
        device = torch::kCUDA;
    } else if (torch::hasMPS() && result.count("cpu") == 0) {
        std::cout << "Using MPS" << std::endl;
        device = torch::kMPS;
    }else{
        std::cout << "Using CPU" << std::endl;
        displayStep = 1;
    }

#ifdef USE_VISUALIZATION
    Visualizer visualizer;
    visualizer.Initialize(numIters);
#endif

    try{
        InputData inputData = inputDataFromX(projectRoot, colmapImageSourcePath, normTranslation, normScale);

        parallel_for(inputData.cameras.begin(), inputData.cameras.end(), [&downScaleFactor](Camera &cam){
            cam.loadImage(downScaleFactor);
        });

        // Split cameras into train/test sets (deterministic)
        std::vector<Camera> trainCams, testCams;
        splitCameras(inputData.cameras, trainCams, testCams, 10);

        // Merge into single list for optimisation loop
        std::vector<Camera> cams = trainCams;
        cams.insert(cams.end(), testCams.begin(), testCams.end());

        Model model(inputData,
                    cams.size(),
                    numDownscales, resolutionSchedule, shDegree, shDegreeInterval, 
                    refineEvery, warmupLength, resetAlphaEvery, densifyGradThresh, densifySizeThresh, stopScreenSizeAt, splitScreenSize,
                    numIters, keepCrs, colorcal,
                    ignoreSaturatedAfter, saturationThreshold,
                    device);

        std::vector< size_t > camIndices( cams.size() );
        std::iota( camIndices.begin(), camIndices.end(), 0 );
        InfiniteRandomIterator<size_t> camsIter( camIndices );

        int imageSize = -1;
        size_t step = 1;

        // Evaluation milestones and snapshot storage
        std::vector<EvalSnapshot> snapshots;
        const std::vector<size_t> milestones = { static_cast<size_t>(numIters) / 10,
                                                  2 * static_cast<size_t>(numIters) / 10,
                                                  3 * static_cast<size_t>(numIters) / 10,
                                                  4 * static_cast<size_t>(numIters) / 10,
                                                  5 * static_cast<size_t>(numIters) / 10,
                                                  6 * static_cast<size_t>(numIters) / 10,
                                                  7 * static_cast<size_t>(numIters) / 10,
                                                  8 * static_cast<size_t>(numIters) / 10,
                                                  9 * static_cast<size_t>(numIters) / 10,
                                                  static_cast<size_t>(numIters) };
        size_t nextMilestoneIdx = 0;

        if (resume != ""){
            step = model.loadPly(resume) + 1;
        }

        for (; step <= numIters; step++){
            Camera& cam = cams[ camsIter.next() ];

            model.optimizersZeroGrad();

            torch::Tensor rgb = model.forward(cam, step);
            torch::Tensor gt = cam.getImage(model.getDownscaleFactor(step));
            gt = gt.to(device);

            torch::Tensor mainLoss = model.mainLoss(rgb, gt, step, ssimWeight);
            mainLoss.backward();
            
            if (step % displayStep == 0) {
                const float percentage = static_cast<float>(step) / numIters;
                std::cout << "Step " << step << ": " << mainLoss.item<float>() << " (" << floor(percentage * 100) << "%)" <<  std::endl;
            }

            if (cam.isTrain){
                model.optimizersStep();
            }else{
                model.stepColourOnly();
            }
            model.schedulersStep(step);
            model.afterTrain(step);

            // Periodic evaluation (catch up if we overshoot)
            while (nextMilestoneIdx < milestones.size() && step >= milestones[nextMilestoneIdx]){
                size_t mIter = milestones[nextMilestoneIdx];
                snapshots.push_back( evaluate(model, trainCams, testCams,
                                              mIter, numIters, ssimWeight) );
                nextMilestoneIdx++;
            }

            if (saveEvery > 0 && step % saveEvery == 0){
                fs::path p(outputScene);
                model.save(p.replace_filename(fs::path(p.stem().string() + "_" + std::to_string(step) + p.extension().string())).string(), step);
            }

            // No separate validation render; evaluation snapshots will handle logging

#ifdef USE_VISUALIZATION
            visualizer.SetInitialGaussianNum(inputData.points.xyz.size(0));
            if (cam.isTrain)
                visualizer.SetLoss(step, mainLoss.item<float>());
            visualizer.SetGaussians(model.means, model.scales, model.featuresDc,
                                    model.opacities);
            visualizer.SetImage(rgb, gt);
            visualizer.Draw();
#endif
        }

        inputData.saveCameras((fs::path(outputScene).parent_path() / "cameras.json").string(), keepCrs, &model);
        model.save(outputScene, numIters);
        // model.saveDebugPly("debug.ply", numIters);

        // Save evaluation snapshots collected during training
        saveEval(snapshots, trainCams, testCams, outputScene);
    }catch(const std::exception &e){
        std::cerr << e.what() << std::endl;
        exit(1);
    }
}
