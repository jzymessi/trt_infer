#ifndef __UTILS_HPP__  
#define __UTILS_HPP__ 

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>

//softmax
template <typename T>
std::vector<float> softmax(const std::vector<T>& x) {
    T max_val = *std::max_element(x.begin(), x.end()); // 找到向量中的最大值
    std::vector<float> exp_x;
    exp_x.reserve(x.size()); // 预分配足够的空间
    for (const T& i : x) {
        exp_x.push_back(std::exp(i - max_val)); // 对每个元素减去最大值并计算 e 的指数
    }
    float sum_exp_x = std::accumulate(exp_x.begin(), exp_x.end(), 0.0f); // 计算所有元素的和
    std::vector<float> y;
    y.reserve(x.size()); // 预分配足够的空间
    for (const float i : exp_x) {
        y.push_back(i / sum_exp_x); // 每个元素除以总和
    }
    return y;
}

void findImagesFiles(const std::string& directory, std::vector<std::string>& imageFiles);
std::vector<float> extractSubVector(const std::vector<float>& source, int startIndex, int endIndex);

#endif