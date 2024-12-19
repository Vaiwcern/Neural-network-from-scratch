#include "ActivationFunction.h"
#include "Kernel.h"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include "Macro.h"

void ReLU::activate(float* input, float* output, int size) const {
    float *d_input, *d_output;

    CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CHECK(cudaMalloc(&d_output, size * sizeof(float)));

    CHECK(cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice));

    int blocks = (size + 255) / 256;
    relu_kernel<<<blocks, 256>>>(d_input, d_output, size);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
}

void ReLU::derivative(float* output, float* d_output, int size) const {
    float *d_out, *d_dout;

    CHECK(cudaMalloc(&d_out, size * sizeof(float)));
    CHECK(cudaMalloc(&d_dout, size * sizeof(float)));

    CHECK(cudaMemcpy(d_out, output, size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dout, d_output, size * sizeof(float), cudaMemcpyHostToDevice));

    int blocks = (size + 255) / 256;
    relu_derivative_kernel<<<blocks, 256>>>(d_out, d_dout, size);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(d_output, d_dout, size * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_dout));
}

void Softmax::activate(float* input, float* output, int size) const {
    float *d_input, *d_output;

    CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CHECK(cudaMalloc(&d_output, size * sizeof(float)));

    CHECK(cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice));

    softmax_kernel<<<1, 1>>>(d_input, d_output, size);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
}

void Softmax::derivative(float* output, float* d_output, int size) const {
    // Softmax + CrossEntropy: derivative trực tiếp = output - target (đã xử lý bên ANN::backward)
    // Không làm gì ở đây
}