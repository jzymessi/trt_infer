#ifndef __FPN_HPP__
#define __FPN_HPP__

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

struct Fpn_Result
{
    int width;
    int height;
    cv::Mat fpn;   //CV_8UC1
};

class FPN : public TRTEngine
{
    public:
        FPN(std::string engine_file_path) : TRTEngine(engine_file_path){}
        ~FPN(){};
        void preprocess(const std::string& image_path , float* blob) override;
        void decode_outputs(std::vector<float>& objects,std::vector<Fpn_Result>& results,int image_size);
        void infer(std::vector<std::string>& file_paths, std::vector<Fpn_Result>& objects);
        std::vector<int> max_index(std::vector<float>& feature_map,int class_num);
    public:
        static const int INPUT_W = 256;
        static const int INPUT_H = 256;
        static const int OUTPUT_W = 64;
        static const int OUTPUT_H = 64;
        static const int channels = 3;
        const char* INPUT_BLOB_NAME = "input";
        const char* OUTPUT_BLOB_NAME = "output";
        const float mean[3] = {123.675, 116.28, 103.53};
        const float std[3] = {58.395, 57.12, 57.375};
        const int batch_size = 16;
        const int class_num = 2;
        float* prob;

};

void FPN::preprocess(const std::string& image_path , float* blob) {
    cv::Mat img_r = cv::imread(image_path);
    cv::Mat img;
    cv::resize(img_r, img, cv::Size(INPUT_W, INPUT_H));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB); //将图像从 BGR 转换为 RGB
    //创建一个大小为图像像素数乘以 3 的浮点数类型的数组，用于存放图像的 blob 数据。img.total() 函数用于获取图像的像素数。
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] = (((float)img.at<cv::Vec3b>(h, w)[c]) - mean[c])/std[c];
            }
        }
    }
}

void max_index_c(std::vector<float>& data, int class_num, int size, std::vector<int>& results) {
  for (int i = 0; i < size; ++i) {
    auto it = std::max_element(data.begin() + i * class_num, data.begin() + (i + 1) * class_num);
    results[i] = std::distance(data.begin() + i * class_num, it);
  }
}

std::vector<int> FPN::max_index(std::vector<float>& feature_map,int class_num) {
    const int size = OUTPUT_W * OUTPUT_H;
    std::vector<int> ret(size);
    max_index_c(feature_map,class_num, size, ret);
    return ret;
}

void FPN::decode_outputs(std::vector<float>& objects,std::vector<Fpn_Result>& results,int image_size) 
{
    int batchSize = OUTPUT_W*OUTPUT_H*2;
    int totalsize = objects.size();
    if(totalsize%batchSize !=0){
        std::cout << "output error! please check ! " << std::endl;
    }
    for(int i=0;i<totalsize;i+=batchSize){
        Fpn_Result result;
        std::vector<float> batch(objects.begin() + i, objects.begin() + i + batchSize);
        std::vector<int> ret = max_index(batch,class_num);
        cv::Mat image = cv::Mat(INPUT_H,INPUT_W,CV_8UC1);
        for(int i=0;i<INPUT_H;++i){
            for(int j=0;j<INPUT_W;++j){
                image.at<uchar>(i, j) = static_cast<uchar>(ret[i * INPUT_W + j]);
            }
        }
        result.width=64;
        result.height=64;
        result.fpn=image;
        results.emplace_back(result);
    }
}

void FPN::infer(std::vector<std::string>& file_paths, std::vector<Fpn_Result>& results){
    int image_size = file_paths.size();
    results.reserve(image_size);
    std::vector<float> objects;
    detect_img(file_paths,objects,INPUT_BLOB_NAME,OUTPUT_BLOB_NAME);
    decode_outputs(objects,results,image_size);
}

void test_fpn_fps(std::string engine_file_path,std::string file_path);

#endif