#include "ActivateFunction.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <limits>
#include <cuda_runtime.h>
#include "CudaHelper.h"

// Phương thức kích hoạt ReLU cho toàn bộ vector
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

// Phương thức kích hoạt Softmax cho toàn bộ vector
void Softmax::activate(float* input, float* output, int size) const {
    // Tính giá trị max(x) để cải thiện độ ổn định số học
    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < size; ++i) {
        max_val = std::max(max_val, input[i]);
    }

    // Tính e^(input[i] - max_val) và tính tổng
    float sum_exp = 0.0f;
    for (int i = 0; i < size; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum_exp += output[i];
    }

    // Chuẩn hóa Softmax (chia cho tổng e^(input[i] - max_val))
    for (int i = 0; i < size; ++i) {
        output[i] /= sum_exp;
    }
}
