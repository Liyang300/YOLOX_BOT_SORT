#include <onnxruntime_cxx_api.h> 
#include "Preprocess.h"
#include "Postprocess.h"
#include "manage.h"
#include "BYTETracker.h"



// ---------- 信号处理 ----------
void signal_handler(int sig) 
{
    std::cout << "\n【收到 Ctrl+C】准备安全退出..." << std::endl;
    _running_flag = false;
    cv_frame.notify_all();
}


void convert_yolox2bytetrack(const std::vector<YOLOX_Detection>& yolox_detections, 
                             std::vector<byte_track::Object>& bytetrack_detections)
{
   bytetrack_detections.clear();
   bytetrack_detections.reserve(yolox_detections.size());
    for (const auto& det : yolox_detections)
    {
        float x = det.x1;
        float y = det.y1;
        float w = det.x2 - det.x1;
        float h = det.y2 - det.y1;
        bytetrack_detections.emplace_back(byte_track::Rect<float>{x, y, w, h}, det.class_id, det.conf);
        //等价于
        // bytetrack_detections.push_back(
        //     byte_track::Object( byte_track::Rect<float>{x,y,w,h}, det.class_id, det.conf )
        // );
    }
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
    byte_track::BYTETracker tracker(30, 30);
    //主循环（显示视频，按 'd' 打印调试信息）
    bool debug_once = false;   // 是否已执行过推理（只执行一次）
    while (_running_flag) 
    {
        std::unique_lock<std::mutex> lock(video_clone_mutex);
        cv_frame.wait(lock, []() { return !frame_queue.empty() || !_running_flag; });
        if (!_running_flag) break;
        if(frame_queue.empty()) continue;

        FrameData data = frame_queue.front();
        frame_queue.pop();
        lock.unlock();

        int key = cv::waitKey(1);

        cv::Mat blob = data.resized_Frame;

        if(model_loaded)
        {
            std::cout << "开始推理..." << std::endl;
            try
            {
                //获取模型形状
                std::vector<int64_t> input_shape = {1, 3, 544, 960};
                size_t input_tensor_size = blob.total();
                //确保内存连续
                if(!blob.isContinuous()) blob = blob.clone();

                Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                        memory_info, blob.ptr<float>(), input_tensor_size,
                        input_shape.data(), input_shape.size()
                );

                //获取输入输出名称

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


                std::vector<byte_track::Object> bytetrack_detections;
                convert_yolox2bytetrack(dets, bytetrack_detections);

                auto tracked = tracker.update(bytetrack_detections);

                cv::Mat vis = data.original_Frame.clone();   // 框是原图坐标，画回原图
                for (const auto& t : tracked)
                {
                    const auto& r  = t->getRect();
                    const int id   = static_cast<int>(t->getTrackId());

                    cv::rectangle(vis,
                        cv::Rect(cvRound(r.x()), cvRound(r.y()), cvRound(r.width()), cvRound(r.height())),
                        cv::Scalar(0, 255, 0), 2);
                    cv::putText(vis, std::to_string(id),
                        cv::Point(cvRound(r.x()), cvRound(r.y()) - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                }
                cv::imshow("ByteTrack", vis);
                cv::waitKey(1);

            }
            catch (const std::exception& e) 
            {
                std::cerr << "推理失败: " << e.what() << std::endl;
            }
        }
        
    }
    
    preprocess.thread_stop();
    std::cout << "main thread exit" << std::endl;
    return 0;
}


