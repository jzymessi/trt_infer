#include "resnet.hpp"
#include "utils.hpp"
#include "fpn.hpp"
#include "yolo.hpp"

int main(int argc, char** argv) {
    const std::string engine_file_path = argv[1];
    const std::string file_path = argv[2];
    // const std::string engine_file_path = "/workspace/tensorrt/models/resnet50/trt/resnet50_b8_fp16.trt";
    // const std::string file_path = "/nfs/algorithm/dataset/public-dataset/imagenet-data/val/";
    test_classification_fps(engine_file_path,file_path);
    return 0;
}