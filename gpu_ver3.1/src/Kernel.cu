#include "Kernel.h"
#include <cmath>
#include <cuda_fp16.h>  // Thêm thư viện để làm việc với half precision

__global__ void forward_kernel(half *input, half *output, half *weights, half *biases, 
                               int input_size, int output_size, int batch_size) {
    // Mỗi block xử lý một sample (b)
    int b = blockIdx.x;
    if (b >= batch_size) return;

    // Mỗi thread xử lý một neuron output (o)
    int o = threadIdx.x;
    if (o >= output_size) return;

    // Shared memory để lưu tile input
    extern __shared__ half s_input[];

    half sum = biases[o];

    // Chia input_size thành từng tile có kích thước = output_size (hoặc nhỏ hơn ở tile cuối)
    for (int i = 0; i < input_size; i += output_size) {
        int idx = i + o;
        half val = __float2half(0.0f);
        if (idx < input_size) {
            val = input[b * input_size + idx];
        }

        // Load giá trị input vào shared memory
        s_input[threadIdx.x] = val;
        __syncthreads();

        int tile_size = min(output_size, input_size - i);

        // Tính dot product trên tile vừa load
        // Mỗi thread xử lý neuron o, nên lấy weights tương ứng: weights[o * input_size + ...]
        for (int k = 0; k < tile_size; ++k) {
            sum = __hadd(sum, __hmul(weights[o * input_size + (i + k)], s_input[k]));
        }
        __syncthreads();
    }

    // Ghi output
    output[b * output_size + o] = sum;
}

__global__ void relu_kernel(half *input, half *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (__hgt(input[idx], __float2half(0.0f))) ? input[idx] : __float2half(0.0f);
    }
}

__global__ void relu_derivative_kernel(half *output, half *d_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        half grad = (__hgt(output[idx], __float2half(0.0f))) ? __float2half(1.0f) : __float2half(0.0f);
        d_output[idx] = __hmul(d_output[idx], grad);
    }
}

__global__ void softmax_kernel(half *input, half *output, int size) {
    // Tính trên 1 vector
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        half max_val = input[0];
        for (int i = 1; i < size; i++) {
            max_val = (__hgt(input[i], max_val)) ? input[i] : max_val;
        }

        half sum_exp = __float2half(0.0f);
        for (int i = 0; i < size; i++) {
            // Chuyển đổi input[i] thành float, tính exp, rồi chuyển lại thành half
            float val = expf(__half2float(input[i]) - __half2float(max_val));
            output[i] = __float2half(val);  // Chuyển kết quả exp trở lại half
            sum_exp = __hadd(sum_exp, output[i]);
        }

        for (int i = 0; i < size; i++) {
            output[i] = __hdiv(output[i], sum_exp);
        }
    }
}


__global__ void update_weights_kernel(half *weights, half *weight_gradients, half *biases, half *bias_gradients, float learning_rate, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        for (int j = 0; j < input_size; ++j) {
            weights[idx * input_size + j] = __hsub(weights[idx * input_size + j], __hmul(__float2half(learning_rate), weight_gradients[idx * input_size + j]));
        }
        biases[idx] = __hsub(biases[idx], __hmul(__float2half(learning_rate), bias_gradients[idx]));
    }
}

__global__ void compute_weight_bias_gradients_kernel(
    half *input, 
    half *output_gradient, 
    half *weights, 
    half *weight_gradients, 
    half *bias_gradients, 
    int input_size, 
    int output_size, 
    int batch_size
) {
    int o = blockIdx.x;
    if (o >= output_size) return;

    int j = threadIdx.x;
    if (j >= input_size) return;

    extern __shared__ half s_weights[];

    if (j < input_size) {
        s_weights[j] = weights[o * input_size + j];
    }
    __syncthreads();

    half wgrad = __float2half(0.0f);

    for (int b = 0; b < batch_size; b++) {
        half grad = output_gradient[b * output_size + o];
        half inp = input[b * input_size + j];
        wgrad = __hadd(wgrad, __hmul(grad, inp));
    }

    if (j < input_size) {
        weight_gradients[o * input_size + j] = wgrad;
    }

    if (j == 0) {
        half total_bgrad = __float2half(0.0f);
        for (int b = 0; b < batch_size; b++) {
            total_bgrad = __hadd(total_bgrad, output_gradient[b * output_size + o]);
        }
        bias_gradients[o] = total_bgrad;
    }
}

__global__ void compute_input_gradient_kernel(
    half *output_gradient, 
    half *weights, 
    half *input_gradient, 
    int input_size, 
    int output_size, 
    int batch_size
) {
    int b = blockIdx.x;
    int j = threadIdx.x;

    if (b >= batch_size || j >= input_size) return;

    half grad_input = __float2half(0.0f);

    for (int o = 0; o < output_size; o++) {
        half w = weights[o * input_size + j];
        half grad = output_gradient[b * output_size + o];
        grad_input = __hadd(grad_input, __hmul(w, grad));
    }

    input_gradient[b * input_size + j] = grad_input;
}
