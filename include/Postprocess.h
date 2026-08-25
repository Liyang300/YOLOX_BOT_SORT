#ifndef POSTPROCESS_H
#define POSTPROCESS_H
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "manage.h"

struct TensorInfo{
    const float* ONNX_data;
    std::vector<int64_t> shape;
};



class Postprocess
{
public:

    Postprocess();
    ~Postprocess();


public:

    std::vector<TensorInfo> convertOrtOutput(const std::vector<Ort::Value>& ort_outputs);

    std::vector<YOLOX_Detection> ONNX_postprocess(const std::vector<TensorInfo>& outputs, 
                                      const cv::Size& original_size,
                                      float conf_threshold,
                                      float nms_threshold);

    
private:
  
};



#endif