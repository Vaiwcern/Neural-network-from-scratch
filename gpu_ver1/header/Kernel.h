#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>

__global__ void backward_kernel(float *input, float *output_gradient, float *weights, float *weight_gradients, float *bias_gradients, int input_size, int output_size);

__global__ void update_weights_kernel(float *weights, float *weight_gradients, float *biases, float *bias_gradients, float learning_rate, int input_size, int output_size);

// Kernel CUDA cho forward pass (tính toán tổng có trọng số mà không có hàm kích hoạt)
__global__ void forward_kernel(float *input, float *output, float *weights, float *biases, int input_size, int output_size);

// Kernel CUDA cho ReLU activation
__global__ void relu_kernel(float *input, float *output, int size);

// Kernel CUDA cho Softmax activation
__global__ void softmax_kernel(float *input, float *output, int size);

#endif