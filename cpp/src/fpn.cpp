#include <string>
#include "fpn.hpp"
#include <vector>
#include <fstream>
#include "utils.hpp"

void test_fpn_fps(std::string engine_file_path,std::string file_path){
    FPN FPN(engine_file_path);
    std::vector<std::string> bmpFiles;
    findImagesFiles(file_path, bmpFiles);
    std::cout << "bmpFiles size: " << bmpFiles.size() << std::endl;
    std::vector<Fpn_Result> results;
    auto start_time = std::chrono::high_resolution_clock::now();
    FPN.infer(bmpFiles,results);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "程序执行时间：" << duration.count() << " 秒" << std::endl;
    std::cout << "图片数量： " << bmpFiles.size() << std::endl;
}
