#include "Kernel.h"
#include <cmath>

__global__ void forward_kernel(float *input, float *output, float *weights, float *biases, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        // Tính toán tổng có trọng số (linear transformation)
        float sum = 0.0f;
        for (int j = 0; j < input_size; ++j) {
            sum += weights[idx * input_size + j] * input[j];
        }
        sum += biases[idx];
        output[idx] = sum;  // Không áp dụng hàm kích hoạt ở đây
    }
}

__global__ void relu_kernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Áp dụng hàm kích hoạt ReLU
        output[idx] = (input[idx] > 0) ? input[idx] : 0;  // ReLU: f(x) = max(0, x)
    }
}

__global__ void softmax_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Tìm giá trị max(x) để cải thiện độ ổn định số học
        float max_val = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < size; ++i) {
            max_val = fmaxf(max_val, input[i]);
        }

        // Tính e^(input[i] - max_val)
        output[idx] = expf(input[idx] - max_val);
    }

    __syncthreads();  // Đồng bộ hóa các thread trong block

    // Tính tổng e^(input[i] - max_val)
    __shared__ float sum_exp;
    if (idx == 0) {
        sum_exp = 0.0f;
        for (int i = 0; i < size; ++i) {
            sum_exp += output[i];
        }
    }

    __syncthreads();  // Đồng bộ hóa sau khi tính tổng

    // Chuẩn hóa Softmax
    if (idx < size) {
        output[idx] /= sum_exp;
    }
}


__global__ void backward_kernel(float *input, float *output_gradient, float *weights, float *weight_gradients, float *bias_gradients, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        for (int j = 0; j < input_size; ++j) {
            weight_gradients[idx * input_size + j] = output_gradient[idx] * input[j];
        }
        bias_gradients[idx] = output_gradient[idx];
    }
}

__global__ void update_weights_kernel(float *weights, float *weight_gradients, float *biases, float *bias_gradients, float learning_rate, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        for (int j = 0; j < input_size; ++j) {
            weights[idx * input_size + j] -= learning_rate * weight_gradients[idx * input_size + j];
        }
        biases[idx] -= learning_rate * bias_gradients[idx];
    }
}

__global__ void cross_entropy_loss_kernel(float* output, float* target, float* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Mỗi thread tính loss cho một phần tử
        float result = -target[idx] * log(output[idx]);  // Tính mất mát cho phần tử idx
        atomicAdd(loss, result);  // Cộng dồn mất mát
    }
}

__global__ void cross_entropy_loss_gradient_kernel(float* output, float* target, float* gradient, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Tính toán gradient của Cross-Entropy Loss
        gradient[idx] = output[idx] - target[idx];
    }
}
