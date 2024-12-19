#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>  // Thư viện hỗ trợ kiểu dữ liệu half

// Forward kernel: tính toán tích trọng số + bias
__global__ void forward_kernel(half *input, half *output, half *weights, half *biases, int input_size, int output_size, int batch_size);

// ReLU activation
__global__ void relu_kernel(half *input, half *output, int size);
// Đạo hàm ReLU
__global__ void relu_derivative_kernel(half *output, half *d_output, int size);

// Softmax activation
__global__ void softmax_kernel(half *input, half *output, int size);

// Backward kernel: tính gradient w.r.t weights và biases, đồng thời tính input_gradient
__global__ void backward_kernel(half *input, half *output_gradient, half *weights, half *weight_gradients, half *bias_gradients, half *input_gradient, int input_size, int output_size, int batch_size);

// Update trọng số
__global__ void update_weights_kernel(half *weights, half *weight_gradients, half *biases, half *bias_gradients, half learning_rate, int input_size, int output_size);

#endif
