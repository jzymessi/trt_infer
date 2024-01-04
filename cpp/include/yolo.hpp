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

struct YOLO_Result
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.1

static void generate_yolo_proposals(std::vector<float>& feat_blob, int output_size, float prob_threshold, std::vector<YOLO_Result>& objects)
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
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;
        float box_objectness = feat_blob[basic_pos+4];
        // std::cout<<*feat_blob<<std::endl;
        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            std::cout << box_prob << " " << prob_threshold << std::endl;
            if (box_prob > prob_threshold)
            {
                YOLO_Result obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        } // class loop
    }

}
static void qsort_descent_inplace(std::vector<YOLO_Result>& faceobjects, int left, int right)
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
static void qsort_descent_inplace(std::vector<YOLO_Result>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static inline float intersection_area(const YOLO_Result& a, const YOLO_Result& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void nms_sorted_bboxes(const std::vector<YOLO_Result>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const YOLO_Result& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const YOLO_Result& b = faceobjects[picked[j]];

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

class YOLO : public TRTEngine
{
    public:
        YOLO(std::string engine_file_path) : TRTEngine(engine_file_path){}
        ~YOLO(){};
        void preprocess(const std::string& image_path , float* blob) override;
        void decode_outputs(std::vector<float>& objects,std::vector<std::vector<YOLO_Result>>& results,int image_size);
        void infer(std::vector<std::string>& file_paths, std::vector<std::vector<YOLO_Result>>& results);
        cv::Mat static_resize(cv::Mat& img);
    public:
        static const int INPUT_W = 640;
        static const int INPUT_H = 640;
        static const int channels = 3;
        const char* INPUT_BLOB_NAME = "images";
        const char* OUTPUT_BLOB_NAME = "output";
        const int batch_size = 1;
        const int class_num = 80;
        std::vector<int> scales;
        

};

cv::Mat YOLO::static_resize(cv::Mat& img) {
    float r = std::min(this->INPUT_W / (img.cols*1.0), this->INPUT_H / (img.rows*1.0));
    scales.push_back(r);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(this->INPUT_W, this->INPUT_H, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

void YOLO::preprocess(const std::string& image_path , float* blob) {
    cv::Mat img_r = cv::imread(image_path);
    cv::Mat pr_img = this->static_resize(img_r);
    cv::cvtColor(pr_img, pr_img, cv::COLOR_BGR2RGB); //将图像从 BGR 转换为 RGB

    // float scale = std::min(this->INPUT_W / (img_r.cols*1.0), this->INPUT_H / (img_r.rows*1.0));

    // int img_h = pr_img.rows;
    // int img_w = pr_img.cols;
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < INPUT_H; h++) 
        {
            for (size_t w = 0; w < INPUT_W; w++) 
            {
                blob[c * INPUT_W * INPUT_H + h * INPUT_W + w] =
                    (((float)pr_img.at<cv::Vec3b>(h, w)[c]) / 255.0f);  //将图像的像素值归一化到 0-1 之间，并赋值给 blob 数组。
            }
        }
    }

}

void YOLO::decode_outputs(std::vector<float>& objects,std::vector<std::vector<YOLO_Result>>& results,int image_size) 
{
    int batchSize = 8400*85;
    int totalsize = objects.size();
    if(totalsize%batchSize !=0){
        std::cout << "output error! please check ! " << std::endl;
    }
    for(int i=0;i<totalsize;i+=batchSize){
        std::vector<YOLO_Result> result;
        std::vector<YOLO_Result> proposals;
        std::vector<float> batch(objects.begin() + i, objects.begin() + i + batchSize);

        generate_yolo_proposals(batch, batchSize, BBOX_CONF_THRESH, proposals);
        std::cout << "num of boxes before nms: " << proposals.size() << std::endl;
        qsort_descent_inplace(proposals);

        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, NMS_THRESH);

        int count = picked.size();
        std::cout << "num of boxes: " << count << std::endl;
        result.resize(count);
        for (int j = 0; j < count; j++)
        {
            result[j] = proposals[picked[j]];

            // adjust offset to original unpadded
            float x0 = (result[j].rect.x) / scales[i];
            float y0 = (result[j].rect.y) / scales[i];
            float x1 = (result[j].rect.x + result[j].rect.width) / scales[i];
            float y1 = (result[j].rect.y + result[j].rect.height) / scales[i];

            // clip
            x0 = std::max(std::min(x0, (float)(INPUT_W - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(INPUT_H - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(INPUT_W - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(INPUT_H - 1)), 0.f);

            result[j].rect.x = x0;
            result[j].rect.y = y0;
            result[j].rect.width = x1 - x0;
            result[j].rect.height = y1 - y0;
        }
        results.emplace_back(result);
    }
    
}

void YOLO::infer(std::vector<std::string>& file_paths, std::vector<std::vector<YOLO_Result>>& results){
    int image_size = file_paths.size();
    results.reserve(image_size);
    std::vector<float> objects;
    detect_img(file_paths,objects,INPUT_BLOB_NAME,OUTPUT_BLOB_NAME);
    decode_outputs(objects,results,image_size);
}

void test_yolo_fps(std::string engine_file_path,std::string file_path);

#endif