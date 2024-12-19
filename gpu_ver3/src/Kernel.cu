#include "Kernel.h"
#include <cmath>


__global__ void forward_kernel(float *input, float *output, float *weights, float *biases, 
                               int input_size, int output_size, int batch_size) {
    // Mỗi block xử lý một sample (b)
    int b = blockIdx.x;
    if (b >= batch_size) return;

    // Mỗi thread xử lý một neuron output (o)
    int o = threadIdx.x;
    if (o >= output_size) return;

    // Shared memory để lưu tile input
    extern __shared__ float s_input[];

    float sum = biases[o];

    // Chia input_size thành từng tile có kích thước = output_size (hoặc nhỏ hơn ở tile cuối)
    for (int i = 0; i < input_size; i += output_size) {
        int idx = i + o;
        float val = 0.0f;
        if (idx < input_size) {
            val = input[b * input_size + idx];
        }

        // Load giá trị input vào shared memory
        s_input[o] = val;
        __syncthreads();

        int tile_size = min(output_size, input_size - i);

        // Tính dot product trên tile vừa load
        // Mỗi thread xử lý neuron o, nên lấy weights tương ứng: weights[o * input_size + ...]
        for (int k = 0; k < tile_size; ++k) {
            sum += weights[o * input_size + (i + k)] * s_input[k];
        }
        __syncthreads();
    }

    // Ghi output
    output[b * output_size + o] = sum;
}


__global__ void backward_kernel(
    float *input, 
    float *output_gradient, 
    float *weights, 
    float *weight_gradients, 
    float *bias_gradients, 
    float *input_gradient,
    int input_size, 
    int output_size, 
    int batch_size
) {
    // Each block handles one output neuron
    int o = blockIdx.x;
    if (o >= output_size) return;

    // Each thread handles one input neuron
    int j = threadIdx.x;
    if (j >= input_size) return;

    // Shared memory for weights of the current output neuron
    extern __shared__ float s_weights[];

    // Load weights into shared memory
    if (j < input_size) {
        s_weights[j] = weights[o * input_size + j];
    }
    __syncthreads();

    float wgrad = 0.0f;

    // Compute weight gradients and accumulate bias gradients
    for (int b = 0; b < batch_size; b++) {
        float grad = output_gradient[b * output_size + o];
        float inp = input[b * input_size + j];
        wgrad += grad * inp;

        // Update input gradients with atomic operations
        float grad_input = s_weights[j] * grad;
        atomicAdd(&input_gradient[b * input_size + j], grad_input);
    }

    // Store the computed weight gradient
    weight_gradients[o * input_size + j] = wgrad;

    // Compute and store bias gradient (only once per output neuron)
    if (j == 0) {
        float total_bgrad = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            total_bgrad += output_gradient[b * output_size + o];
        }
        atomicAdd(&bias_gradients[o], total_bgrad);
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
    // Tính trên 1 vector
    if (threadIdx.x == 0 && blockIdx.x == 0) {
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


__global__ void update_weights_kernel(float *weights, float *weight_gradients, float *biases, float *bias_gradients, float learning_rate, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Kiểm tra nếu thread đang xử lý idx hợp lệ
    if (idx < output_size) {
        // Sử dụng shared memory để lưu trữ gradient tạm thời trong mỗi block
        extern __shared__ float shared_weights[];

        // Đảm bảo rằng không vượt quá kích thước shared memory
        // Tối ưu cho kích thước nhỏ hơn hoặc bằng 256
        float* shared_gradients = shared_weights + input_size; 

        // Tải dữ liệu vào shared memory, mỗi thread sẽ xử lý một phần trọng số và gradient
        for (int j = threadIdx.x; j < input_size; j += blockDim.x) {
            shared_weights[j] = weights[idx * input_size + j];  // Cập nhật trọng số trong shared memory
            shared_gradients[j] = weight_gradients[idx * input_size + j];  // Cập nhật gradient trong shared memory
        }

        // Đồng bộ hóa threads trong block để đảm bảo tất cả dữ liệu đã được tải vào shared memory
        __syncthreads();

        // Cập nhật trọng số và gradient trong shared memory
        for (int j = 0; j < input_size; ++j) {
            shared_weights[j] -= learning_rate * shared_gradients[j];
        }

        // Cập nhật lại trọng số trong global memory
        for (int j = threadIdx.x; j < input_size; j += blockDim.x) {
            weights[idx * input_size + j] = shared_weights[j];  // Lưu lại trọng số đã cập nhật vào global memory
        }

        // Cập nhật bias trong global memory
        if (threadIdx.x == 0) {  // Chỉ một thread trong block thực hiện cập nhật bias
            biases[idx] -= learning_rate * bias_gradients[idx];
        }
    }
}
