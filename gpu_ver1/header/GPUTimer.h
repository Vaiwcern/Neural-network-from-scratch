#ifndef GPU_TIMER_H
#define GPU_TIMER_H

#include <cuda_runtime.h>  // Thêm thư viện CUDA runtime

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer();
    ~GpuTimer();

    void Start();
    void Stop();
    float Elapsed();
};

#endif // GPU_TIMER_H
