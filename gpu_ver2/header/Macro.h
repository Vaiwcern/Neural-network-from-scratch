#ifndef MACRO_H
#define MACRO_H

#include <cuda_runtime.h>
#include <iostream>

// Macro để kiểm tra lỗi CUDA
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", ";         \
        std::cerr << "code: " << error << ", reason: "                         \
                  << cudaGetErrorString(error) << std::endl;                   \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

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

#endif // MACRO_H