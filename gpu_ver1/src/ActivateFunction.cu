#include "ActivateFunction.h"
#include <cmath>
#include <cuda_runtime.h>

// Phương thức kích hoạt ReLU cho toàn bộ vector
void ReLU::activate(float* input, float* output, int size) const {
    float *d_input, *d_output;

    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (size + 255) / 256;
    relu_kernel<<<blocks, 256>>>(d_input, d_output, size);
    
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

// Phương thức kích hoạt Softmax cho toàn bộ vector
void Softmax::activate(float* input, float* output, int size) const {
    float *d_input, *d_output;

    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (size + 255) / 256;
    softmax_kernel<<<blocks, 256>>>(d_input, d_output, size);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
