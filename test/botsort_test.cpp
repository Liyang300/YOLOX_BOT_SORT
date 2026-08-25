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






// ---------- 主函数 ----------
int main() 
{
    signal(SIGINT, signal_handler);

    //启动预处理线程（读取视频，生成 FrameData）
    Preprocess  preprocess;
    Postprocess postprocess;

    std::string video_path = "/mnt/e/pyCode/数据集/2.mp4";
    preprocess.thread_start(video_path);

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
        // 确保 yolox.onnx 在可执行文件同目录，或使用绝对路径
        session = std::make_unique<Ort::Session>(env, "/home/yaoyi/APPs/video/demo1/yolox_voc_s_no_decode.onnx", session_options);
        model_loaded = true;
        std::cout << "ONNX 模型加载成功！" << std::endl;

        // 打印模型输入/输出信息
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = session->GetInputCount();
        size_t num_outputs = session->GetOutputCount();
        std::cout << "输入数量: " << num_inputs << ", 输出数量: " << num_outputs << std::endl;
{
        // for (size_t i = 0; i < num_inputs; ++i) 
        // {
        //     auto name = session->GetInputNameAllocated(i, allocator);
        //     std::cout << "输入 " << i << " 名称: " << name.get() << std::endl;
        //     auto type_info = session->GetInputTypeInfo(i);
        //     auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        //     auto shape = tensor_info.GetShape();
        //     std::cout << "  形状: ";
        //     for (auto d : shape) std::cout << d << " ";
        //     std::cout << std::endl;
        // }
        // for (size_t i = 0; i < num_outputs; ++i) 
        // {
        //     auto name = session->GetOutputNameAllocated(i, allocator);
        //     std::cout << "输出 " << i << " 名称: " << name.get() << std::endl;
        //     auto type_info = session->GetOutputTypeInfo(i);
        //     auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        //     auto shape = tensor_info.GetShape();
        //     std::cout << "  形状: ";
        //     for (auto d : shape) std::cout << d << " ";
        //     std::cout << std::endl;
        // }
}

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
        std::unique_lock<std::mutex> lock(video_clone_mutex);
        cv_frame.wait(lock, []() { return !frame_queue.empty() || !_running_flag; });
        if (!_running_flag) break;
        if (frame_queue.empty()) continue;

        FrameData data = frame_queue.front();
        frame_queue.pop();
        lock.unlock();

        // 显示原始帧
        //cv::imshow("ORG BGR", data.original_Frame);

        int key = cv::waitKey(1);
        //std::cout << "waitKey returned: " << key << std::endl;
        // 按 'd' 键打印调试信息
        // if (key == 'd' || key == 'D') 
        // {
            cv::Mat blob = data.resized_Frame;   // 形状 [1, 3, 544, 960], CV_32F
            // std::cout << "\n========== 当前帧 Blob 信息 ==========" << std::endl;
            // std::cout << "形状: " << blob.size << std::endl;
            // std::cout << "数据类型: " << blob.type() << " (CV_32F = 5)" << std::endl;
            // std::cout << "总元素: " << blob.total() << std::endl;
            // std::cout << "是否连续: " << (blob.isContinuous() ? "是" : "否") << std::endl;

            // if (blob.type() == CV_32F) 
            // {
            //     double minVal, maxVal;
            //     cv::minMaxLoc(blob, &minVal, &maxVal);
            //     cv::Scalar mean, stddev;
            //     cv::meanStdDev(blob, mean, stddev);
            //     std::cout << "最小值: " << minVal << std::endl;
            //     std::cout << "最大值: " << maxVal << std::endl;
            //     std::cout << "均值: " << mean[0] << std::endl;
            //     std::cout << "标准差: " << stddev[0] << std::endl;
            // }

            // if (blob.total() >= 10 && blob.type() == CV_32F) 
            // {
            //     std::cout << "前 10 个数值: ";
            //     const float* ptr = blob.ptr<float>();
            //     for (int i = 0; i < 10; ++i) std::cout << ptr[i] << " ";
            //     std::cout << std::endl;
            // }

            // 如果模型已加载且尚未执行推理，则执行一次并打印输出
            if (model_loaded) 
            {
                std::cout << "\n执行一次 ONNX Runtime 推理以获取输出形状..." << std::endl;
                try 
                {
                    // 准备输入
                    // 输入形状从模型读取（这里硬编码，但可以从模型获取）
                    std::vector<int64_t> input_shape = {1, 3, 544, 960};  // 根据训练配置


                    size_t input_tensor_size = blob.total();
                    // 确保内存连续
                    if (!blob.isContinuous()) blob = blob.clone();
                    Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
                    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                        memory_info, blob.ptr<float>(), input_tensor_size,
                        input_shape.data(), input_shape.size()
                    );

                    // 获取输入/输出名称
                    Ort::AllocatorWithDefaultOptions allocator;
                    std::vector<std::string> input_names_str;
                    for (size_t i = 0; i < session->GetInputCount(); ++i) {
                        input_names_str.push_back(session->GetInputNameAllocated(i, allocator).get());
                    }
                    std::vector<const char*> input_names;
                    for (const auto& name : input_names_str) {
                        input_names.push_back(name.c_str());
                    }

                    // 输出名称
                    std::vector<std::string> output_names_str;
                    for (size_t i = 0; i < session->GetOutputCount(); ++i) {
                        output_names_str.push_back(session->GetOutputNameAllocated(i, allocator).get());
                    }
                    std::vector<const char*> output_names;
                    for (const auto& name : output_names_str) {
                        output_names.push_back(name.c_str());
                    }


                    // 推理
                    auto outputs = session->Run(Ort::RunOptions{nullptr},
                                                input_names.data(), &input_tensor, 1,
                                                output_names.data(), output_names.size());

                    
                    cv::Size orig_size(data.original_Frame.cols, data.original_Frame.rows);

                  

                    std::vector<TensorInfo> output_tran = postprocess.convertOrtOutput(outputs);

                    std::vector<YOLOX_Detection>  dets = postprocess.ONNX_postprocess(output_tran, orig_size, 0.4, 0.4);


                    std::cout << "Before convert, dets.size() = " << dets.size() << std::endl;
                    std::vector<Detection> bot_dets = convert_to_tracker_format(dets);
                    std::cout << "After convert, bot_dets.size() = " << bot_dets.size() << std::endl;

                    auto tracks = tracker.track(bot_dets, data.original_Frame); 


                    cv::Mat vis = data.original_Frame.clone();
                    for(const auto& det : dets)
                    {
                        cv::rectangle(vis, cv::Point2f(det.x1, det.y1), 
                                      cv::Point2f(det.x2, det.y2), cv::Scalar(0, 255, 0), 2);
                        char text[64];
                        snprintf(text, sizeof(text), "%.2f", det.conf);
                        cv::putText(vis, text, cv::Point2f(det.x1, det.y1 - 5),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

                    }

                    //遍历所有跟踪轨迹
                    for (const auto& track : tracks) 
                    {
                        // 获取边界框 [x, y, width, height]（左上角坐标）
                        std::vector<float> tlwh = track->get_tlwh();
                        int x = static_cast<int>(tlwh[0]);
                        int y = static_cast<int>(tlwh[1]);
                        int w = static_cast<int>(tlwh[2]);
                        int h = static_cast<int>(tlwh[3]);

                        // 绘制矩形（每个 ID 用不同颜色，可以加个随机颜色生成器，这里先写固定绿）
                        cv::rectangle(vis, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
                        
                        // 在框上方绘制 ID 和置信度（置信度可从 track 里拿，但 track 存的是历史平滑值，直接用也行）
                        std::string label = "ID:" + std::to_string(track->track_id);
                        cv::putText(vis, label, cv::Point(x, y - 10),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                    }
                    cv::imshow("detection", vis);
{
                    // for (size_t i = 0; i < outputs.size(); ++i) 
                    // {
                    //     auto& output = outputs[i];
                    //     auto type_info = output.GetTensorTypeAndShapeInfo();
                    //     auto shape = type_info.GetShape();
                    //     std::cout << "输出 " << i << " 形状: ";
                    //     for (auto d : shape) std::cout << d << " ";
                    //     std::cout << std::endl;
                    //     // 打印前几个值
                    //     if (type_info.GetElementCount() >= 5) 
                    //     {
                    //         const float* ptr = output.GetTensorData<float>();
                    //         std::cout << "  前5个值: ";
                    //         for (int j = 0; j < 5; ++j) std::cout << ptr[j] << " ";
                    //         std::cout << std::endl;
                    //     }
                    // }
}

                } 
                catch (const std::exception& e) 
                {
                    std::cerr << "推理失败: " << e.what() << std::endl;
                }
                debug_once = true;   // 只执行一次
            } 
            else if (!model_loaded) 
            {
                std::cout << "模型未加载，跳过推理。" << std::endl;
            }
            std::cout << "=======================================\n" << std::endl;
        //}

        // 退出键
        if (key == 'q' || key == 27) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // 清理
    preprocess.thread_stop();
    std::cout << "main thread exit" << std::endl;
    return 0;
}