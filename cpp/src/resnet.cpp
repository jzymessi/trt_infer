#include <string>
#include "resnet.hpp"
#include <vector>
#include <fstream>
#include "utils.hpp"

struct ImageInfo {
    std::string path;
    std::string id;
};

void test_classification_fps(std::string engine_file_path,std::string file_path){
    Resnet Resnet(engine_file_path);
    std::vector<std::string> bmpFiles;
    findImagesFiles(file_path, bmpFiles);
    // std::cout << bmpFiles[0] << std::endl;
    std::vector<Classification_Result> results;
    auto start_time = std::chrono::high_resolution_clock::now();
    Resnet.infer(bmpFiles,results);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    float time_s = float(duration.count()/1000000);
    std::cout << "程序执行时间：" << time_s << " 秒" << std::endl;
    std::cout << "图片数量： " << bmpFiles.size() << std::endl;
    std::cout << "FPS: " << float(bmpFiles.size()/time_s) << std::endl;
}


void test_classification_accuracy(std::string engine_file_path,std::string file_path){
    Resnet Resnet(engine_file_path);
    std::ifstream infile(file_path);
    std::vector<ImageInfo> image_infos;
    std::string line;
    while (std::getline(infile, line)) {
        auto pos = line.find(' ');
        auto path = line.substr(0, pos);
        auto id = line.substr(pos + 1);
        image_infos.push_back({path, id});
    }
    infile.close();

    std::vector<std::string> file_paths;
    for(int i =0; i < image_infos.size(); i++ )
    {
        file_paths.emplace_back(image_infos[i].path);
    }

    std::cout << "file_paths len: " << file_paths.size() << std::endl;
    int image_size = image_infos.size();
    std::vector<Classification_Result> results;
    results.reserve(image_size);
    Resnet.infer(file_paths,results);
    int one_hit_count  = 0;
    std::cout << results.size() << std::endl;
    for(size_t i=0;i<results.size();i++)
    {
        std::cout<< image_infos[i].path << std::endl;
        if(results[i].label == std::stoi(image_infos[i].id))
        {
            one_hit_count++;
        }   
    }
    float accuracy = 0.0;
    accuracy  =  float(one_hit_count) / image_infos.size();
    accuracy   = accuracy  * 100;
    std::cout <<" accuracy: "<< accuracy <<"%" << std::endl;

}