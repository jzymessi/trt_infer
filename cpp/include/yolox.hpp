#ifndef __YOLO_HPP__
#define __YOLO_HPP__

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
struct box{
    float x0,y0,x1,y1;
};

struct YOLOX_Result
{
    int label;
    float prob;
    box bbox;
};

#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.1

static void generate_yolo_proposals(std::vector<float>& feat_blob, int output_size, float prob_threshold, std::vector<YOLOX_Result>& objects)
{
    const int num_class = 80;
    auto dets = output_size / (num_class + 5);
    for (int boxs_idx = 0; boxs_idx < dets; boxs_idx++)
    {
        const int basic_pos = boxs_idx *(num_class + 5);
        float x_center = feat_blob[basic_pos+0];
        float y_center = feat_blob[basic_pos+1];
        float w = feat_blob[basic_pos+2];
        float h = feat_blob[basic_pos+3];
        // std::cout << x_center << " " << y_center << " " << w << " " << h << std::endl;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;
        float x1 = x_center + w * 0.5f;
        float y1 = y_center + h * 0.5f;

        // std::cout << x0 << " " << y0 << " " << x1 << " " << y1 << std::endl;
        float box_objectness = feat_blob[basic_pos+4];
        // std::cout<<*feat_blob<<std::endl;
        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            // std::cout << box_prob << " " << prob_threshold << std::endl;
            if (box_prob > prob_threshold)
            {
                YOLOX_Result obj;
                obj.bbox.x0 = x0;
                obj.bbox.y0 = y0;
                obj.bbox.x1 = x1;
                obj.bbox.y1 = y1;
                obj.label = class_idx;
                obj.prob = box_prob;
                // std::cout << x0 << " " << y0 << " " << w << " " << h << " " << box_prob << " " << class_idx << std::endl;

                objects.push_back(obj);
            }

        } // class loop
    }

}
static void qsort_descent_inplace(std::vector<YOLOX_Result>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}
static void qsort_descent_inplace(std::vector<YOLOX_Result>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static inline float intersection_area(const YOLOX_Result& a, const YOLOX_Result& b)
{
    float x_left = std::max(a.bbox.x0, b.bbox.x0);
    float y_top = std::max(a.bbox.y0, b.bbox.y0);
    float x_right = std::min(a.bbox.x1, b.bbox.x1);
    float y_bottom = std::min(a.bbox.y1, b.bbox.y1);

    // 计算交集矩形的宽度和高度
    float width = std::max(0.0f, x_right - x_left);
    float height = std::max(0.0f, y_bottom - y_top);

    // 计算交集面积并返回
    return width * height;
}

static void nms_sorted_bboxes(const std::vector<YOLOX_Result>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        // areas[i] = faceobjects[i].rect.area();
        areas[i] = (faceobjects[i].bbox.x1 - faceobjects[i].bbox.x0) * (faceobjects[i].bbox.y1 - faceobjects[i].bbox.y0);

    }

    for (int i = 0; i < n; i++)
    {
        const YOLOX_Result& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const YOLOX_Result& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

class YOLOX : public TRTEngine
{
    public:
        YOLOX(std::string engine_file_path) : TRTEngine(engine_file_path){}
        ~YOLOX(){};
        void preprocess(const std::string& image_path , float* blob) override;
        void decode_outputs(std::vector<float>& objects,std::vector<std::vector<YOLOX_Result>>& results,int image_size);
        void infer(std::vector<std::string>& file_paths, std::vector<std::vector<YOLOX_Result>>& results);
        cv::Mat static_resize(cv::Mat& img);
    public:
        static const int INPUT_W = 640;
        static const int INPUT_H = 640;
        static const int channels = 3;
        const char* INPUT_BLOB_NAME = "images";
        const char* OUTPUT_BLOB_NAME = "output";
        const int batch_size = 1;
        const int class_num = 80;
        std::vector<float> scales;
        std::vector<std::vector<int>> img_w_h;
        

};

cv::Mat YOLOX::static_resize(cv::Mat& img) {
    float r = std::min(this->INPUT_W / (img.cols*1.0), this->INPUT_H / (img.rows*1.0));
    // std::cout << "resize scale: " << r << std::endl;
    scales.push_back(r);
    img_w_h.push_back({img.cols, img.rows});
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(this->INPUT_W, this->INPUT_H, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

void YOLOX::preprocess(const std::string& image_path , float* blob) {
    cv::Mat img_r = cv::imread(image_path);
    cv::Mat pr_img = this->static_resize(img_r);
    // cv::cvtColor(pr_img, pr_img, cv::COLOR_BGR2RGB); //将图像从 BGR 转换为 RGB

    // float scale = std::min(this->INPUT_W / (img_r.cols*1.0), this->INPUT_H / (img_r.rows*1.0));

    int img_h = pr_img.rows;
    int img_w = pr_img.cols;

    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                if(c * img_w * img_h + h * img_w + w < input_h*input_h*channel*batch_size )
                {
                    blob[c * img_w * img_h + h * img_w + w] =(((float)pr_img.at<cv::Vec3b>(h, w)[c])); 
                    // std::cout << "blob[" << c * img_w * img_h + h * img_w + w << "] = " << ((float)pr_img.at<cv::Vec3b>(h, w)[c]) << std::endl;
                }
                else{
                    std::cout << "Error: Index out of bounds!" << std::endl;
                }
            }
        }
    }

}

void YOLOX::decode_outputs(std::vector<float>& objects,std::vector<std::vector<YOLOX_Result>>& results,int image_size) 
{
    int batchSize = 8400*85;
    int totalsize = objects.size();
    if(totalsize%batchSize !=0){
        std::cout << "output error! please check ! " << std::endl;
    }
    for(int i=0;i<totalsize;i+=batchSize){
        std::vector<YOLOX_Result> result;
        std::vector<YOLOX_Result> proposals;
        std::vector<float> batch(objects.begin() + i, objects.begin() + i + batchSize);

        generate_yolo_proposals(batch, batchSize, BBOX_CONF_THRESH, proposals);
        // std::cout << "num of boxes before nms: " << proposals.size() << std::endl;
        qsort_descent_inplace(proposals);

        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, NMS_THRESH);

        int count = picked.size();
        // std::cout << "num of boxes: " << count << std::endl;
        result.resize(count);
        for (int j = 0; j < count; j++)
        {
            result[j] = proposals[picked[j]];

            // adjust offset to original unpadded
            // std::cout << scales[i] << std::endl;
            float x0 = (result[j].bbox.x0) / scales[i];
            float y0 = (result[j].bbox.y0) / scales[i];
            float x1 = (result[j].bbox.x1) / scales[i];
            float y1 = (result[j].bbox.y1) / scales[i];

            // clip
            x0 = std::max(std::min(x0, (float)(img_w_h[i][0] - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_w_h[i][1] - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w_h[i][0] - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_w_h[i][1] - 1)), 0.f);

            result[j].bbox.x0 = x0;
            result[j].bbox.y0 = y0;
            result[j].bbox.x1 = x1;
            result[j].bbox.y1 = y1;
        }
        results.emplace_back(result);
    }
    
}

void YOLOX::infer(std::vector<std::string>& file_paths, std::vector<std::vector<YOLOX_Result>>& results){
    int image_size = file_paths.size();
    results.reserve(image_size);
    std::vector<float> objects;
    detect_img(file_paths,objects,INPUT_BLOB_NAME,OUTPUT_BLOB_NAME);
    decode_outputs(objects,results,image_size);
}

void test_yolox_fps(std::string engine_file_path,std::string file_path);
void test_yolox_draw_results(std::string engine_file_path,std::string file_path,std::string output_image_path);
#endif