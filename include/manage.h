#ifndef MANAGE_H
#define MANAGE_H

#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <signal.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <memory>



#define YOLO_W  960
#define YOLO_H  544





struct FrameData{
    cv::Mat original_Frame;
    cv::Mat resized_Frame;
};

struct YOLOX_Detection{
    int class_id;
    float x1, x2, y1, y2;
    float conf;
};

//std::vector<cv::Mat> original_Frame1;





extern std::atomic<bool> _running_flag;
extern std::condition_variable cv_frame;
extern std::mutex video_clone_mutex;
extern std::queue<FrameData> frame_queue;



#endif