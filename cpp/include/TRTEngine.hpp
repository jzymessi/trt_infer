#ifndef __TRTENGINE_HPP__
#define __TRTENGINE_HPP__

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
#include <omp.h>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

using namespace nvinfer1;
static Logger gLogger;

class TRTEngine
{
    public:
        TRTEngine(std::string engine_file_path);
        ~TRTEngine();
        virtual void preprocess(const std::string& image_path , float* blob) = 0;
        void doInference(IExecutionContext& context, float* input, float* output, const int output_size, int size,const char* INPUT_BLOB_NAME, const char* OUTPUT_BLOB_NAME);
        void save_outputs(float* prob, int output_size, std::vector<float>& objects);
        void detect_img(std::vector<std::string>& file_paths, std::vector<float>& objects,const char* INPUT_BLOB_NAME, const char* OUTPUT_BLOB_NAME);
        int batch_size = 0;
        int channel = 0;
        int input_w = 0;
        int input_h = 0;
    private:
        float* prob;
        ICudaEngine* engine;
        IRuntime* runtime;
        IExecutionContext* context;
        int output_size = 1;
        // int _get_batch = 0;

};


TRTEngine::TRTEngine(std::string engine_file_path)
{
    size_t size{0};
    char *trtModelStream{nullptr};
    //使用二进制方式读取文件，将其存放到字符数组trtModelStream中。
    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good()) //判断文件是否成功打开，如果成功打开则执行后续操作
    {
        file.seekg(0, file.end); //将文件指针移到文件的结尾，这里是为了获取文件的大小。
        size = file.tellg();    //获取文件的大小，并将其保存到变量 size 中。
        file.seekg(0, file.beg);  //将文件指针重新指向文件的开头,这里是为了读取文件内容.
        trtModelStream = new char[size]; //创建一个字符数组（char数组），大小为文件的大小。
        assert(trtModelStream);  //确保创建字符数组成功，如果创建失败程序将终止。
        file.read(trtModelStream, size);  //读取文件的内容到字符数组中。
        file.close();
    }
    // std::cout << "engine init finished" << std::endl;
    //创建一个 TensorRT 的运行时（runtime）实例，以便在 TensorRT 中执行推理（inference）
    gLogger.setReportableSeverity(ILogger::Severity::kERROR);  //配置logger的级别 kINFO kWARNING 
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr); //确保 runtime 变量不为空指针 
    //使用 runtime 中的 deserializeCudaEngine 函数，将字符数组中的内容反序列化为一个 ICudaEngine 对象。
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    //创建一个执行上下文（execution context）
    context = engine->createExecutionContext(); 
    assert(context != nullptr);
    delete[] trtModelStream; //释放字符数组的内存空间。
    // get batch_size,channel,input_w,input_h from model 
    auto input_dims = engine->getBindingDimensions(0);  
    int input_size = input_dims.nbDims; 
    batch_size = input_dims.d[0];
    channel = input_dims.d[1];
    input_w = input_dims.d[2];
    input_h = input_dims.d[3];
    //engine 的 getBindingDimensions 方法，获取引擎的第 1 个输出张量的维度信息，并将其赋值给变量 out_dims。
    auto out_dims = engine->getBindingDimensions(1);
    //遍历 out_dims 中的每一个维度，并将其相乘，得到输出张量的大小。
    for(int j=0;j<out_dims.nbDims;j++) {
        // std::cout << "ouput size: " << j  << " " << out_dims.d[j] << std::endl;
        this->output_size *= out_dims.d[j];
    }
    //创建一个大小为 output_size 的 float 类型的数组，用于存放推理结果。
    this->prob = new float[this->output_size];
}

TRTEngine::~TRTEngine()
{
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();   
}

void TRTEngine::doInference(IExecutionContext& context, float* input, float* output, const int output_size, int size,const char* INPUT_BLOB_NAME, const char* OUTPUT_BLOB_NAME) {
    //获取与推断上下文相关联的CUDA引擎。 context对象是IExecutionContext接口的一个实例，它表示用于在CUDA引擎上执行推断的执行上下文的实例。
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);  // 1 input and 1 output
    void* buffers[2]; //创建一个大小为 2 的 void 类型的指针数组，用于存放输入和输出张量的 GPU 缓冲区。

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);//获取了输入张量在TensorRT引擎中的绑定索引
    // std::cout << "inputIndex: " << inputIndex << std::endl;
    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT); //检查输入张量的数据类型是否为 float 类型
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);//获取了输出张量在TensorRT引擎中的绑定索引
    // std::cout << "outputIndex: " << outputIndex << std::endl;
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT); //检查输出张量的数据类型是否为 float 类型
    int mBatchSize = engine.getMaxBatchSize(); //获取引擎的最大批量大小
    // std::cout << "mBatchSize: " << mBatchSize << std::endl;
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], size * sizeof(float))); //分配输入张量的 GPU 缓冲区
    CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float))); //分配输出张量的 GPU 缓冲区

    // Create stream
    //创建一个 CUDA 流，用于异步执行 CUDA 操作, 可以在不等待先前提交的操作完成的情况下，继续提交新的操作。
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, size * sizeof(float), cudaMemcpyHostToDevice, stream));
    //使用输入和输出缓冲区以及 CUDA 流在执行上下文中异步执行推理。
    context.enqueue(1, buffers, stream, nullptr);
    //使用异步内存拷贝函数cudaMemcpyAsync，将设备上的输出数据output从设备内存复制到主机上的输出缓冲区buffers[outputIndex]中。
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream); //等待所有的异步操作完成

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

void TRTEngine::save_outputs(float* prob, int output_size, std::vector<float>& objects) 
{
    #pragma omp parallel for
    for (int i = 0; i < output_size; i++) 
    {
        objects.push_back(prob[i]);
    }
}

void TRTEngine::detect_img(std::vector<std::string>& file_paths, std::vector<float>& objects,const char* INPUT_BLOB_NAME, const char* OUTPUT_BLOB_NAME)
{   
    int image_size = file_paths.size();
    float* blob = new float[input_w * input_h * channel * batch_size];

    // 为每个batch分配一个新的blob
    float** blobs = new float*[batch_size];
    #pragma omp parallel for
    for (int batch = 0; batch < batch_size; batch++) {
        blobs[batch] = new float[channel * input_w * input_h];
    }

    int num_batches = image_size / batch_size;
    int remaining_images = image_size % batch_size;
    
    #pragma omp parallel for
    for (int batch = 0; batch < num_batches; batch++) {
        // 将图片放入blobs中，这里假设有一个函数loadImageToBlob，可以将图片数据加载到blobs中
        for (int i = 0; i < batch_size; i++) {
            int image_index = batch * batch_size + i;
            preprocess(file_paths[image_index], blobs[i]);
        }
        //将blobs合并到一个float*中
        for (int batch = 0; batch < batch_size; batch++) {
            memcpy(blob + batch * channel * input_w * input_h, blobs[batch], channel * input_w * input_h * sizeof(float));
        }
        

        // run inference
        int size = batch_size * channel * input_w * input_h;
        doInference(*context, blob, this->prob, output_size, size,INPUT_BLOB_NAME, OUTPUT_BLOB_NAME); //执行推理 
        // decode output
        save_outputs(this->prob, this->output_size, objects);
        
    }
    // 填充剩余的部分
    if (remaining_images > 0) {
        // 将剩余的图片放入blobs中
        // #pragma omp parallel for
        for (int i = 0; i < remaining_images; i++) {
            int image_index = num_batches * batch_size + i;
            preprocess(file_paths[image_index], blobs[i]);
        }
        #pragma omp parallel for
        //将blobs合并到一个float*中
        for (int batch = 0; batch < batch_size; batch++) {
            memcpy(blob + batch * channel * input_w * input_h, blobs[batch], channel * input_w * input_h * sizeof(float));
        }
        // run inference
        int size = batch_size * channel * input_w * input_h;
        doInference(*context, blob, this->prob, output_size, size,INPUT_BLOB_NAME, OUTPUT_BLOB_NAME); //执行推理 
        // decode output
        save_outputs(this->prob, this->output_size, objects);

    }

    //free blobs
    #pragma omp parallel for
    for (int batch = 0; batch < batch_size; batch++) {
            delete[] blobs[batch];
    }
    delete blob;

}

#endif
