#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>

// Forward kernel: tính toán tích trọng số + bias
__global__ void forward_kernel(float *input, float *output, float *weights, float *biases, int input_size, int output_size, int batch_size);


// ReLU activation
__global__ void relu_kernel(float *input, float *output, int size);
// Đạo hàm ReLU
__global__ void relu_derivative_kernel(float *output, float *d_output, int size);

// Softmax activation
__global__ void softmax_kernel(float *input, float *output, int size);


// Backward kernel: tính gradient w.r.t weights và biases, đồng thời tính input_gradient
__global__ void backward_kernel(float *input, float *output_gradient, float *weights, float *weight_gradients, float *bias_gradients, float *input_gradient, int input_size, int output_size, int batch_size);


// Update trọng số
__global__ void update_weights_kernel(float *weights, float *weight_gradients, float *biases, float *bias_gradients, float learning_rate, int input_size, int output_size);

#endif
