#include <onnxruntime_cxx_api.h> 
#include "Preprocess.h"
#include "Postprocess.h"
#include "manage.h"
#include "DataType.h"
#include "BoTSORT.h"


// ---------- 信号处理 ----------
void signal_handler(int sig) 
{
    std::cout << "\n【收到 Ctrl+C】准备安全退出..." << std::endl;
    _running_flag = false;
    cv_frame.notify_all();
}

std::vector<Detection> convert_to_tracker_format(const std::vector<YOLOX_Detection>& yolox_detections)
{
    std::vector<Detection> Dets;
    Dets.reserve(yolox_detections.size());

    for(const auto& yolo_detection : yolox_detections)
    {
        Detection Det;
        Det.bbox_tlwh = cv::Rect_(yolo_detection.x1, 
                                  yolo_detection.y1,
                                  yolo_detection.x2 - yolo_detection.x1,
                                  yolo_detection.y2 - yolo_detection.y1);
        Det.class_id = yolo_detection.class_id;
        Det.confidence = yolo_detection.conf;
        Dets.push_back(Det);
    }
    return Dets;
}

int main()
{
    signal(SIGINT, signal_handler);

    //启动预处理线程（读取视频，生成 FrameData）
    Preprocess  preprocess;
    Postprocess postprocess;

    std::string vedio_path = "/mnt/e/pyCode/数据集/2.mp4";
    preprocess.thread_start(vedio_path);

    //初始化 ONNX Runtime 环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOX");
    Ort::SessionOptions session_options;
    //session_options.SetIntraOpNumThreads(4);

    session_options.SetIntraOpNumThreads(6);
    session_options.SetInterOpNumThreads(2);

    // 开启全部图优化
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // 禁用不需要算子、CPU专用优化
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);


    // 如果 GPU 可用，可添加：session_options.AppendExecutionProvider_CUDA(0);


    std::unique_ptr<Ort::Session> session;
    bool model_loaded = false;
    try
    {
        //添加yolox.onnx模型路径
        session = std::make_unique<Ort::Session>(env, "/home/yaoyi/APPs/video/demo1/yolox_voc_s_no_decode.onnx", session_options);
        model_loaded = true;
        std::cout << "ONNX 模型加载成功！" << std::endl;

        //打印模型形状
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session->GetInputCount();
        size_t num_output_nodes = session->GetOutputCount();
        std::cout<<"输入节点数："<<num_input_nodes<<" 输出节点数："<<num_output_nodes<<std::endl;
    }
    catch (const std::exception& e) 
    {
        std::cerr << "加载 ONNX 模型失败: " << e.what() << std::endl;
        std::cerr << "请检查文件路径或文件完整性。" << std::endl;
        // 即使模型加载失败，程序仍继续运行，以便查看视频预处理效果
    }

    // 初始化跟踪器
    BoTSORT tracker("../config/tracker.ini", "../config/gmc.ini");

    //主循环（显示视频，按 'd' 打印调试信息）
    bool debug_once = false;   // 是否已执行过推理（只执行一次）
    while (_running_flag) 
    {
        
    }
}


