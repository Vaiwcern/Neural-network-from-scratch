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

void Softmax::activate(float* input, float* output, int size) const {
    // Tạo bộ nhớ trên device
    float* d_input;
    float* d_output;

    CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CHECK(cudaMalloc(&d_output, size * sizeof(float)));

    // Sao chép dữ liệu từ host vào device
    CHECK(cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice));

    // Thực thi kernel
    int block_size = 256;  // Kích thước block
    int num_blocks = (size + block_size - 1) / block_size;  // Tính số block

    softmax_kernel<<<num_blocks, block_size>>>(d_input, d_output, size);
    CHECK(cudaDeviceSynchronize());  // Đồng bộ hóa để đảm bảo kernel đã hoàn thành

    // Sao chép kết quả từ device về host
    CHECK(cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));

    // Giải phóng bộ nhớ trên device
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
}

