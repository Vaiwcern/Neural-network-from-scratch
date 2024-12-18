#include "DenseLayer.h"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include "Kernel.h"

DenseLayer::DenseLayer(int input_size, int output_size, ActivationFunction* activation)
    : input_size(input_size), output_size(output_size), activation(activation)
{
    // Cấp phát bộ nhớ cho weights và biases
    weights = new float[input_size * output_size];
    biases = new float[output_size];

    weight_gradients = new float[input_size * output_size];
    bias_gradients = new float[output_size];

    // last_input, last_output sẽ lưu cho cả batch (tối đa), 
    // tuy nhiên ta không biết batch_size tối đa, có thể cấp phát tối đa 
    // hoặc cấp phát động khi forward. Giả sử batch_size nhỏ, ta cấp phát lớn hơn.
    // Ở đây giả sử batch_size tối đa không vượt quá train. Hoặc dùng new khi forward.
    // Để đơn giản, ta sẽ mỗi lần forward, backward cấp phát lại động (nhưng để code đơn giản, ta giữ static)
    // Tạm giả sử batch_size tối đa là 1024:
    int max_batch = 1024;
    last_input = new float[input_size * max_batch];
    last_output = new float[output_size * max_batch];

    srand((unsigned int)time(0));
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            weights[i * input_size + j] = ((float)rand() / RAND_MAX) * 0.01f;  // nhỏ để ổn định
        }
        biases[i] = 0.0f;
    }
}

// Phương thức forward pass (batch)
void DenseLayer::forward(float* input, float* output, int batch_size) {
    // Lưu input để backward
    memcpy(last_input, input, input_size * batch_size * sizeof(float));

    float *d_input, *d_output, *d_weights, *d_biases;

    // Cấp phát bộ nhớ trên device (GPU)
    cudaMalloc(&d_input, input_size * batch_size * sizeof(float));
    cudaMalloc(&d_output, output_size * batch_size * sizeof(float));
    cudaMalloc(&d_weights, input_size * output_size * sizeof(float));
    cudaMalloc(&d_biases, output_size * sizeof(float));

    // Sao chép dữ liệu từ host (CPU) vào device (GPU)
    cudaMemcpy(d_input, input, input_size * batch_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Tính toán forward pass (W*x + b) cho batch_size mẫu
    // Số thread: mỗi thread xử lý 1 neuron output cho 1 mẫu => total = output_size * batch_size
    int total_threads = output_size * batch_size;
    int blocks = (total_threads + 255) / 256;
    forward_kernel<<<blocks, 256>>>(d_input, d_output, d_weights, d_biases, input_size, output_size, batch_size);
    cudaDeviceSynchronize();

    // Lấy giá trị linear về host để apply activation
    float* linear_output = new float[output_size * batch_size];
    cudaMemcpy(linear_output, d_output, output_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Áp dụng hàm kích hoạt (ReLU hoặc Softmax) cho từng mẫu
    // Với softmax, cần thực hiện cho từng vector output_size, batch_size lần.
    // Ta có thể gọi activation->activate trên từng mẫu một.
    // Để tối ưu, có thể viết kernel softmax cho batch, nhưng ở đây ta lặp trên CPU.
    for (int b = 0; b < batch_size; b++) {
        activation->activate(&linear_output[b * output_size], &output[b * output_size], output_size);
    }

    // Lưu output để backward
    memcpy(last_output, output, output_size * batch_size * sizeof(float));

    // Giải phóng
    delete[] linear_output;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(d_biases);
}

// Phương thức backward pass (tính toán gradient cho batch)
// output_gradient: (batch_size * output_size)
// input_gradient: (batch_size * input_size)
void DenseLayer::backward(float* output_gradient, float* input_gradient, int batch_size) {
    // Áp dụng đạo hàm activation
    // Đạo hàm activation trên batch
    float* d_act = new float[output_size * batch_size];
    memcpy(d_act, output_gradient, output_size * batch_size * sizeof(float));

    // activation->derivative áp dụng trên từng vector output_size
    for (int b = 0; b < batch_size; b++) {
        activation->derivative(&last_output[b * output_size], &d_act[b * output_size], output_size);
    }

    // Tính gradient weights, biases, input_gradient qua kernel
    float *d_input, *d_act_dev, *d_weights, *d_wgrad, *d_bgrad, *d_igrad;

    cudaMalloc(&d_input, input_size * batch_size * sizeof(float));
    cudaMalloc(&d_act_dev, output_size * batch_size * sizeof(float));
    cudaMalloc(&d_weights, input_size * output_size * sizeof(float));
    cudaMalloc(&d_wgrad, input_size * output_size * sizeof(float));
    cudaMalloc(&d_bgrad, output_size * sizeof(float));
    cudaMalloc(&d_igrad, input_size * batch_size * sizeof(float));

    cudaMemcpy(d_input, last_input, input_size * batch_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_act_dev, d_act, output_size * batch_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Set gradient ban đầu =0
    cudaMemset(d_wgrad, 0, input_size * output_size * sizeof(float));
    cudaMemset(d_bgrad, 0, output_size * sizeof(float));
    cudaMemset(d_igrad, 0, input_size * batch_size * sizeof(float));

    int total_threads = output_size * batch_size;
    int blocks = (total_threads + 255) / 256;
    backward_kernel<<<blocks, 256>>>(d_input, d_act_dev, d_weights, d_wgrad, d_bgrad, d_igrad, input_size, output_size, batch_size);
    cudaDeviceSynchronize();

    // Copy gradient về host
    // Giờ đây weight_gradients là tổng gradient trên toàn batch. Bias_gradients cũng vậy.
    // backward_kernel đã atomicAdd và tính gộp. Chúng ta sẽ chia ở chỗ backward (hoặc đã chia trong output_gradient).
    cudaMemcpy(weight_gradients, d_wgrad, input_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(bias_gradients, d_bgrad, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    if (input_gradient) {
        cudaMemcpy(input_gradient, d_igrad, input_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_input);
    cudaFree(d_act_dev);
    cudaFree(d_weights);
    cudaFree(d_wgrad);
    cudaFree(d_bgrad);
    cudaFree(d_igrad);

    delete[] d_act;
}

// Phương thức cập nhật trọng số
void DenseLayer::update_weights(float learning_rate, int batch_size) {
    float *d_weights, *d_weight_gradients, *d_biases, *d_bias_gradients;

    cudaMalloc(&d_weights, input_size * output_size * sizeof(float));
    cudaMalloc(&d_weight_gradients, input_size * output_size * sizeof(float));
    cudaMalloc(&d_biases, output_size * sizeof(float));
    cudaMalloc(&d_bias_gradients, output_size * sizeof(float));

    cudaMemcpy(d_weights, weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_gradients, weight_gradients, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_gradients, bias_gradients, output_size * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (output_size + 255) / 256;
    update_weights_kernel<<<blocks, 256>>>(d_weights, d_weight_gradients, d_biases, d_bias_gradients, learning_rate, input_size, output_size);
    cudaDeviceSynchronize();

    cudaMemcpy(weights, d_weights, input_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(biases, d_biases, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_weights);
    cudaFree(d_weight_gradients);
    cudaFree(d_biases);
    cudaFree(d_bias_gradients);
}

DenseLayer::~DenseLayer() {
    delete[] weights;
    delete[] biases;
    delete[] weight_gradients;
    delete[] bias_gradients;
    delete[] last_input;
    delete[] last_output;
    delete activation;
}
