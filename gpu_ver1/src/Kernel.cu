#include "Kernel.h"
#include <cmath>

__global__ void forward_kernel(float *input, float *output, float *weights, float *biases, int input_size, int output_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // idx đại diện cho một neuron output của một mẫu trong batch
    // output_size * b + o
    if (idx < output_size * batch_size) {
        int b = idx / output_size; // batch index
        int o = idx % output_size; // output neuron index

        float sum = biases[o];
        for (int j = 0; j < input_size; ++j) {
            sum += weights[o * input_size + j] * input[b * input_size + j];
        }
        output[idx] = sum;  // Linear output
    }
}

__global__ void relu_kernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > 0) ? input[idx] : 0;
    }
}

__global__ void relu_derivative_kernel(float *output, float *d_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float grad = (output[idx] > 0) ? 1.0f : 0.0f;
        d_output[idx] *= grad;
    }
}

__global__ void softmax_kernel(float *input, float *output, int size) {
    if (threadIdx.x == 0) {
        float max_val = -INFINITY;
        for (int i = 0; i < size; i++) {
            if (input[i] > max_val) max_val = input[i];
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < size; i++) {
            float val = expf(input[i] - max_val);
            output[i] = val;
            sum_exp += val;
        }

        for (int i = 0; i < size; i++) {
            output[i] /= sum_exp;
        }
    }
}

__global__ void backward_kernel(float *input, float *output_gradient, float *weights, 
                                float *weight_gradients, float *bias_gradients, float *input_gradient,
                                int input_size, int output_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size * batch_size) {
        int b = idx / output_size; // batch index
        int o = idx % output_size; // output neuron index

        float grad = output_gradient[idx]; // delta cho neuron o của mẫu b

        // Cập nhật bias_gradients (cần atomic do nhiều thread cùng ghi)
        atomicAdd(&bias_gradients[o], grad);

        // weight_gradients và input_gradient cũng phải atomicAdd
        for (int j = 0; j < input_size; ++j) {
            float wg = grad * input[b * input_size + j];
            atomicAdd(&weight_gradients[o * input_size + j], wg);
            atomicAdd(&input_gradient[b * input_size + j], weights[o * input_size + j] * grad);
        }
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
