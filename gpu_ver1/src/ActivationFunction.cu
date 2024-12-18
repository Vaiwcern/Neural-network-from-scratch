#include "ActivationFunction.h"
#include "Kernel.h"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

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

void ReLU::derivative(float* output, float* d_output, int size) const {
    // output ở đây là output sau ReLU
    float *d_out, *d_dout;
    cudaMalloc(&d_out, size * sizeof(float));
    cudaMalloc(&d_dout, size * sizeof(float));

    cudaMemcpy(d_out, output, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dout, d_output, size * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (size + 255) / 256;
    relu_derivative_kernel<<<blocks,256>>>(d_out, d_dout, size);
    cudaDeviceSynchronize();

    cudaMemcpy(d_output, d_dout, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_dout);
}

// Phương thức kích hoạt Softmax cho toàn bộ vector
void Softmax::activate(float* input, float* output, int size) const {
    float *d_input, *d_output;

    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    softmax_kernel<<<1, 1>>>(d_input, d_output, size);
    cudaDeviceSynchronize();

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

void Softmax::derivative(float* output, float* d_output, int size) const {
    // Với Softmax + CrossEntropy, derivative thường được tính trực tiếp: d_output = output - target
    // Ở đây ta không cần kernel riêng.
    // Chỉ để tuân thủ interface
    // Không làm gì ở đây.
}