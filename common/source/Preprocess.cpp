#include "Preprocess.h"
#include "manage.h"

std::atomic<bool> _running_flag(true);
std::condition_variable cv_frame;
std::mutex  video_clone_mutex;
std::queue<FrameData> frame_queue;



Preprocess::Preprocess()
{
   
}
Preprocess::~Preprocess()
{
    _running_flag = false;

}
void Preprocess::thread_start(const std::string& video_path)
{
    _running_flag = true;
    _work = std::thread(&Preprocess::video_read_thread, this, video_path);
    std::cout << "Preprocess thread start" << std::endl;
}


void Preprocess::thread_stop()
{
    _running_flag = false;
    cv_frame.notify_all();
    if(_work.joinable())
    {
        _work.join();
    }
}


bool Preprocess::video_read_thread(const std::string video_path)
{
    cv::VideoCapture cap(video_path);
    if(!cap.isOpened())
    {
        std::cerr << "can't open video" << video_path << std::endl;
        return false;
    }
    
    
    int srcW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int srcH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
 
    cv::Mat frame, rgbInput, bgrResize;


    auto fps_t0 = std::chrono::steady_clock::now();
    int frame_cnt = 0;


    while(cap.read(frame) && _running_flag)
    {
        frame_cnt++;
         // 每秒打印一次解码真实FPS
        auto now = std::chrono::steady_clock::now();
        double delta = std::chrono::duration<double>(now - fps_t0).count();
        if(delta >= 1.0)
        {
            std::cout << "【预处理解码FPS】" << frame_cnt << std::endl;
            frame_cnt = 0;
            fps_t0 = now;
        }

        if(frame.empty())
        {
            break;
        }


        cv::resize(frame, bgrResize, cv::Size(YOLO_W, YOLO_H));

        //cv::cvtColor(bgrResize, rgbInput, cv::COLOR_BGR2RGB);

        cv::Mat rgb_float;
        bgrResize.convertTo(rgb_float, CV_32F);

        // // 添加均值/标准差归一化
        // cv::Scalar mean = {0.485f, 0.456f, 0.406f};
        // cv::Scalar std  = {0.229f, 0.224f, 0.225f};
        // cv::subtract(rgb_float, mean, rgb_float);
        // cv::divide(rgb_float, std, rgb_float);

        int h = rgb_float.rows;
        int w = rgb_float.cols;

        cv::Mat blob(1, 3 * h * w, CV_32F);
        std::vector<cv::Mat> channels(3);

        cv::split(rgb_float, channels);

        float* blob_ptr = blob.ptr<float>(0);

        
        for(int c = 0; c < 3; c++)
        {
            float* dst = blob_ptr + c * (h * w);
            const float* src = channels[c].ptr<float>(0);
            memcpy(dst, src, h * w * sizeof(float));
        }

        FrameData data;
        data.original_Frame = frame.clone();
        data.resized_Frame = blob.clone();
        {
            std::lock_guard<std::mutex> lock(video_clone_mutex);
            frame_queue.push(data);
            if(frame_queue.size() > 10)
            {
                frame_queue.pop();
            }
        }

      
        
        cv_frame.notify_one();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    cv_frame .notify_all();
    cap.release();
    cv::destroyAllWindows();
    std::cout << "end" << std::endl;

    return true;
}

