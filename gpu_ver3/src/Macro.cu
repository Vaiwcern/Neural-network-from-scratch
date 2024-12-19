#include "Macro.h"

// Định nghĩa các hàm thành viên của GpuTimer
GpuTimer::GpuTimer()
{
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
}

GpuTimer::~GpuTimer()
{
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
}

void GpuTimer::Start()
{
    CHECK(cudaEventRecord(start, 0));
    CHECK(cudaEventSynchronize(start));
}

void GpuTimer::Stop()
{
    CHECK(cudaEventRecord(stop, 0));
}

float GpuTimer::Elapsed()
{
    float elapsed;
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    return elapsed;
}
