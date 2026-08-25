#ifndef PREPROCESS_H
#define PREPROCESS_H


#include <thread>
#include "manage.h"




class Preprocess
{

public:


public:
    Preprocess();
    ~Preprocess();
    
    void thread_start(const std::string& video_path);
    void thread_stop();
    bool video_read_thread(const std::string video_path);

private:
    
    std::thread _work;

};






#endif 