#include <string>
#include "yolox.hpp"
#include <vector>
#include <fstream>
#include "utils.hpp"

void test_yolox_fps(std::string engine_file_path,std::string file_path){
    YOLOX YOLOX(engine_file_path);
    std::vector<std::string> bmpFiles;
    findImagesFiles(file_path, bmpFiles);
    std::cout << "bmpFiles size: " << bmpFiles.size() << std::endl;
    std::vector<std::vector<YOLOX_Result>> results;
    auto start_time = std::chrono::high_resolution_clock::now();
    YOLOX.infer(bmpFiles,results);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "程序执行时间：" << duration.count() << " 微秒" << std::endl;
    std::cout << "图片数量： " << bmpFiles.size() << std::endl;
}

void test_yolox_draw_results(std::string engine_file_path,std::string file_path,std::string output_image_path){
    YOLOX YOLOX(engine_file_path);
    std::vector<std::string> bmpFiles;
    findImagesFiles(file_path, bmpFiles);
    std::vector<std::vector<YOLOX_Result>> results;
    YOLOX.infer(bmpFiles,results);
    // std::cout << "results size: " << results.size() << std::endl;
    for(int i = 0; i < results.size(); i++){
        cv::Mat img = cv::imread(bmpFiles[i]);
        for(int j = 0; j < results[i].size(); j++){
            // std::cout << "label: " << results[i][j].label << " prob: " << results[i][j].prob << " rect: " << results[i][j].bbox.x0 << " " << results[i][j].bbox.y0 << " " << results[i][j].bbox.x1 << " " << results[i][j].bbox.y1 << std::endl;
            if (results[i][j].prob > 0.5){
                cv::rectangle(img, cv::Rect(results[i][j].bbox.x0, results[i][j].bbox.y0, results[i][j].bbox.x1 - results[i][j].bbox.x0, results[i][j].bbox.y1 - results[i][j].bbox.y0), cv::Scalar(0, 255, 0), 2);
                cv::putText(img, results[i][j].label + " : " + std::to_string(results[i][j].prob), cv::Point(results[i][j].bbox.x0, results[i][j].bbox.y0 + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
            
    }
    output_image_path = output_image_path + "/" + bmpFiles[i].substr(bmpFiles[i].find_last_of("/") + 1);
    cv::imwrite(output_image_path, img);
    }
}
