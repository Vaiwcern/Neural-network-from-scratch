#include "ActivationFunction.h"
#include "Kernel.h"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include "Macro.h"
#include <cuda_fp16.h>  // Thêm thư viện để làm việc với half precision

void ReLU::activate(half* input, half* output, int size) const {
    half *d_input, *d_output;

    CHECK(cudaMalloc(&d_input, size * sizeof(half)));
    CHECK(cudaMalloc(&d_output, size * sizeof(half)));

    // Chuyển từ half sang float trước khi sao chép vào bộ nhớ của GPU
    float* temp_input = new float[size];
    for (int i = 0; i < size; ++i) {
        temp_input[i] = __half2float(input[i]);
    }
    CHECK(cudaMemcpy(d_input, temp_input, size * sizeof(float), cudaMemcpyHostToDevice));
    delete[] temp_input;

    int blocks = (size + 255) / 256;
    relu_kernel<<<blocks, 256>>>(d_input, d_output, size);
    CHECK(cudaDeviceSynchronize());

    // Chuyển lại từ half sang float khi sao chép kết quả về host
    float* temp_output = new float[size];
    CHECK(cudaMemcpy(temp_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; ++i) {
        output[i] = __float2half(temp_output[i]);
    }
    delete[] temp_output;

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
}

void ReLU::derivative(half* output, half* d_output, int size) const {
    half *d_out, *d_dout;

    CHECK(cudaMalloc(&d_out, size * sizeof(half)));
    CHECK(cudaMalloc(&d_dout, size * sizeof(half)));

    // Chuyển từ half sang float trước khi sao chép vào bộ nhớ của GPU
    float* temp_out = new float[size];
    float* temp_dout = new float[size];
    for (int i = 0; i < size; ++i) {
        temp_out[i] = __half2float(output[i]);
        temp_dout[i] = __half2float(d_output[i]);
    }
    CHECK(cudaMemcpy(d_out, temp_out, size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dout, temp_dout, size * sizeof(float), cudaMemcpyHostToDevice));
    delete[] temp_out;
    delete[] temp_dout;

    int blocks = (size + 255) / 256;
    relu_derivative_kernel<<<blocks, 256>>>(d_out, d_dout, size);
    CHECK(cudaDeviceSynchronize());

    // Chuyển lại từ half sang float khi sao chép kết quả về host
    CHECK(cudaMemcpy(temp_dout, d_dout, size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; ++i) {
        d_output[i] = __float2half(temp_dout[i]);
    }
    delete[] temp_dout;

    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_dout));
}

void Softmax::activate(half* input, half* output, int size) const {
    half *d_input, *d_output;

    CHECK(cudaMalloc(&d_input, size * sizeof(half)));
    CHECK(cudaMalloc(&d_output, size * sizeof(half)));

    // Chuyển từ half sang float trước khi sao chép vào bộ nhớ của GPU
    float* temp_input = new float[size];
    for (int i = 0; i < size; ++i) {
        temp_input[i] = __half2float(input[i]);
    }
    CHECK(cudaMemcpy(d_input, temp_input, size * sizeof(float), cudaMemcpyHostToDevice));
    delete[] temp_input;

    softmax_kernel<<<1, 1>>>(d_input, d_output, size);
    CHECK(cudaDeviceSynchronize());

    // Chuyển lại từ half sang float khi sao chép kết quả về host
    float* temp_output = new float[size];
    CHECK(cudaMemcpy(temp_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; ++i) {
        output[i] = __float2half(temp_output[i]);
    }
    delete[] temp_output;

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
}

void Softmax::derivative(half* output, half* d_output, int size) const {
    // Softmax + CrossEntropy: derivative trực tiếp = output - target (đã xử lý bên ANN::backward)
    // Không làm gì ở đây
}
