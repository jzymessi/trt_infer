#ifndef __RESNET_HPP__
#define __RESNET_HPP__

#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.hpp"
#include <cmath>
#include <algorithm>
#include "TRTEngine.hpp"
#include "utils.hpp"

struct Classification_Result
{
    int label;
    float score;
};

class Resnet : public TRTEngine
{
    public:
        Resnet(std::string engine_file_path) : TRTEngine(engine_file_path){}
        ~Resnet(){};
        void preprocess(const std::string& image_path , float* blob) override;
        void decode_outputs(std::vector<float>& objects,std::vector<Classification_Result>& results,int image_size);
        void infer(std::vector<std::string>& file_paths, std::vector<Classification_Result>& objects);
    public:
        const char* INPUT_BLOB_NAME = "input";
        const char* OUTPUT_BLOB_NAME = "output";
        const float mean[3] = {123.675, 116.28, 103.53};
        const float std[3] = {58.395, 57.12, 57.375};
        // const int batch_size = 4;
        float* prob;

};

void Resnet::preprocess(const std::string& image_path , float* blob) {
    cv::Mat img_r = cv::imread(image_path);
    cv::Mat img;
    cv::resize(img_r, img, cv::Size(224, 224));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB); //将图像从 BGR 转换为 RGB
    //创建一个大小为图像像素数乘以 3 的浮点数类型的数组，用于存放图像的 blob 数据。img.total() 函数用于获取图像的像素数。
    // int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channel; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                // 检查是否越界
                if(c * img_w * img_h + h * img_w + w < input_h*input_h*channel*batch_size ){
                    blob[c * img_w * img_h + h * img_w + w] = (((float)img.at<cv::Vec3b>(h, w)[c]) - mean[c])/std[c];
                }
                else{
                    std::cout << "Error: Index out of bounds!" << std::endl;
                }
            }
        }
    }
}

void Resnet::decode_outputs(std::vector<float>& objects,std::vector<Classification_Result>& results,int image_size) 
{
    #pragma omp parallel for
    for(size_t i=0;i<image_size*3;i += 3){
        Classification_Result result;
        std::vector<float> temp;
        temp.push_back(objects[i]);
        temp.push_back(objects[i+1]);
        temp.push_back(objects[i+2]);
        std::vector<float> y = softmax(temp);
        auto max_iter = std::max_element(y.begin(), y.end());
        int max_index = std::distance(y.begin(), max_iter);
        result.label = max_index;
        result.score = *max_iter;
        results.push_back(result);
    }
}

void Resnet::infer(std::vector<std::string>& file_paths, std::vector<Classification_Result>& results){
    int image_size = file_paths.size();
    results.reserve(image_size);
    std::vector<float> objects;
    // std::cout << "111" << std::endl;
    detect_img(file_paths,objects,INPUT_BLOB_NAME,OUTPUT_BLOB_NAME);
    // std::cout << "222" << std::endl;
    decode_outputs(objects,results,image_size);
}

void test_classification_fps(std::string engine_file_path,std::string file_path);
void test_classification_accuracy(std::string engine_file_path,std::string file_path);

#endif