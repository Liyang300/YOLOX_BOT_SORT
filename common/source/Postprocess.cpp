#include "Postprocess.h"



Postprocess::Postprocess()
{

}

Postprocess::~Postprocess()
{

}



// 输入：onnxruntime推理原始输出
// 输出：自定义TensorInfo结构体数组，给后处理使用
std::vector<TensorInfo> Postprocess::convertOrtOutput(const std::vector<Ort::Value>& ort_outputs)
{
    std::vector<TensorInfo> out_info_list;
    out_info_list.reserve(ort_outputs.size());

    for (const auto& tensor_val : ort_outputs)
    {
        TensorInfo info;
        // auto shape_info = tensor_val.GetTensorTypeAndShapeInfo();
        // info.shape = shape_info.GetShape();
        // size_t elem_count = shape_info.GetElementCount();
        // info.ONNX_data.resize(elem_count);
        // const float* src = tensor_val.GetTensorData<float>();
        // std::copy(src, src + elem_count, info.ONNX_data.begin());
        // out_info_list.push_back(std::move(info));
        info.ONNX_data = tensor_val.GetTensorData<float>();
        auto shape_info = tensor_val.GetTensorTypeAndShapeInfo();
        info.shape = shape_info.GetShape();
        out_info_list.push_back(std::move(info));
    }

    return out_info_list;
}

std::vector<YOLOX_Detection> Postprocess::ONNX_postprocess(const std::vector<TensorInfo>& outputs, 
                                                const cv::Size& original_size,
                                                float conf_threshold,
                                                float nms_threshold)
{
    struct StrideInfo{
        int stride;
        float h, w;
    };

    std::vector<StrideInfo> infos = {
        {8, 68, 120},
        {16, 34, 60},
        {32, 17, 30},
    };

    std::vector<YOLOX_Detection> detections;
    for(size_t idx = 0; idx < outputs.size(); ++idx)
    {
        const auto& output   = outputs[idx];
        const auto shape     = output.shape;

        if(shape.size() != 4) continue;
        int batch    = static_cast<int>(shape[0]);
        int channels = static_cast<int>(shape[1]);
        int H        = static_cast<int>(shape[2]);
        int W        = static_cast<int>(shape[3]);
        if(channels != 6) continue;
        auto ONNX_data = output.ONNX_data;

        int stride = infos[idx].stride;
        //===========================================================
        // 在 ONNX_postprocess 函数的第一个 stride 层中，循环前添加：
        // if (idx == 0) {
        //     for (int c = 0; c < 6; ++c) {
        //         float sum = 0.0f;
        //         for (int i = 0; i < H * W; ++i) {
        //             sum += ONNX_data[c * H * W + i];
        //         }
        //         std::cout << "Channel " << c << " mean: " << sum / (H * W) << std::endl;
        //     }
        // }



        std::vector<float> gride_x(H * W), gride_y(H * W);
        for(int h = 0; h < H; ++h)
        {
            for(int w = 0; w < W; ++w)
            {
                int index = h * W + w;
                gride_x[index] = static_cast<float>(w);
                gride_y[index] = static_cast<float>(h);
            }
        }
        for(int b = 0; b<batch; ++b)
        {
            for(int i = 0; i < H * W; ++i)
            {
                float reg_x = ONNX_data[(b * channels + 0) * (H * W) + i];
                float reg_y = ONNX_data[(b * channels + 1) * (H * W) + i];
                float reg_w = ONNX_data[(b * channels + 2) * (H * W) + i];
                float reg_h = ONNX_data[(b * channels + 3) * (H * W) + i];
                float obj   = ONNX_data[(b * channels + 4) * (H * W) + i];
                float cls   = ONNX_data[(b * channels + 5) * (H * W) + i];

                //sigmoid 激活
                float obj_sigmoid = 1.0f / (1.0f + exp(-obj));
                float cls_sigmoid = 1.0f / (1.0f + exp(-cls));
                float conf = obj_sigmoid * cls_sigmoid;

                static int total_anchors = 0;
                static float max_conf = 0.0f;
                total_anchors++;
                if (conf > max_conf) max_conf = conf;
                // 在每个 stride 层级结束时打印（或在所有循环结束后）
                // 在循环内打印前几个 anchor 的值：
                // if (i < 5 && idx == 0) 
                // {
                //     std::cout << "Anchor " << i << ": obj=" << obj << ", cls=" << cls << ", conf=" << conf << std::endl;
                // }

                if(conf < conf_threshold) continue;

                //解码坐标
                float cx = (reg_x + gride_x[i]) * stride;
                float cy = (reg_y + gride_y[i]) * stride;
                float w_box = expf(reg_w) * stride;
                float h_box = expf(reg_h) * stride;

                //转换为输入尺寸的坐标
                float x1 = cx - w_box / 2.0f;
                float y1 = cy - h_box / 2.0f;
                float x2 = cx + w_box / 2.0f;
                float y2 = cy + h_box / 2.0f;

                float scale_x = static_cast<float>(original_size.width)  / static_cast<float>(YOLO_W);
                float scale_y = static_cast<float>(original_size.height) / static_cast<float>(YOLO_H);
                x1 *= scale_x;
                y1 *= scale_y;
                x2 *= scale_x;
                y2 *= scale_y;

                YOLOX_Detection det;
                det.x1 = x1;
                det.y1 = y1;
                det.x2 = x2;
                det.y2 = y2;
                det.conf = conf;
                det.class_id = 0;
                detections.push_back(det);
                //std::cout << "Total anchors processed: " << total_anchors << ", max_conf=" << max_conf << std::endl;
            }
        }
    }

    //NMS
    std::sort(detections.begin(), detections.end(), 
              [](const YOLOX_Detection& a, const YOLOX_Detection& b) {return a.conf > b.conf;});
    std::vector<YOLOX_Detection> keep;
    std::vector<bool> removed(detections.size(), false);

    for(size_t i = 0; i < detections.size(); ++i)
    {
        if(removed[i]) continue;
        keep.push_back(detections[i]);
        for(size_t j = i + 1; j < detections.size(); ++j)
        {
            if(removed[j]) continue;
            float x1_i = std::max(detections[i].x1, detections[j].x1);
            float y1_i = std::max(detections[i].y1, detections[j].y1);
            float x2_i = std::min(detections[i].x2, detections[j].x2);
            float y2_i = std::min(detections[i].y2, detections[j].y2);
            float w_i  = std::max(0.0f, x2_i - x1_i);
            float h_i  = std::max(0.0f, y2_i - y1_i);
            float inter = w_i * h_i;
            float area1 = (detections[i].x2 - detections[i].x1) * (detections[i].y2 - detections[i].y1);
            float area2 = (detections[j].x2 - detections[j].x1) * (detections[j].y2 - detections[j].y1);
            float iou = inter / (area1 + area2 - inter);
            if(iou > nms_threshold) removed[j] = true;
        }
    }
    return keep;
}