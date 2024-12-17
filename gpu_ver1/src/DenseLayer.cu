#include "DenseLayer.h"
#include <cstdlib>
#include <ctime>

// Constructor
DenseLayer::DenseLayer(int input_size, int output_size, ActivationFunction* activation)
    : input_size(input_size), output_size(output_size), activation(activation) 
{
    // Cấp phát bộ nhớ cho weights và biases
    weights = new float[input_size * output_size];
    biases = new float[output_size];

    // Khởi tạo ngẫu nhiên trọng số và độ chệch
    srand(time(0));
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            weights[i * input_size + j] = ((float)rand() / RAND_MAX) * 2 - 1;  // Random between -1 and 1
        }
        biases[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
}

// Phương thức forward pass
void DenseLayer::forward(float* input, float* output) {
    float *d_input, *d_output, *d_weights, *d_biases;

    // Cấp phát bộ nhớ trên device (GPU)
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMalloc(&d_weights, input_size * output_size * sizeof(float));
    cudaMalloc(&d_biases, output_size * sizeof(float));

    // Sao chép dữ liệu từ host (CPU) vào device (GPU)
    cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Tính toán forward pass (tính tổng có trọng số và độ chệch)
    int blocks = (output_size + 255) / 256;
    forward_kernel<<<blocks, 256>>>(d_input, d_output, d_weights, d_biases, input_size, output_size);
    cudaDeviceSynchronize();  // Đồng bộ hóa để đảm bảo kernel đã hoàn thành

    // Áp dụng hàm kích hoạt (ReLU hoặc Softmax)
    activation->activate(d_output, d_output, output_size);

    // Sao chép kết quả từ device về host
    cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Giải phóng bộ nhớ trên device
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(d_biases);
}

// Phương thức backward pass (tính toán gradient)
void DenseLayer::backward(float* input, float* output_gradient, float* weight_gradients, float* bias_gradients, int batch_size) {
    float *d_input, *d_output_gradient, *d_weights, *d_weight_gradients, *d_bias_gradients;

    // Cấp phát bộ nhớ trên device (GPU)
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output_gradient, output_size * sizeof(float));
    cudaMalloc(&d_weights, input_size * output_size * sizeof(float));
    cudaMalloc(&d_weight_gradients, input_size * output_size * sizeof(float));
    cudaMalloc(&d_bias_gradients, output_size * sizeof(float));

    // Sao chép dữ liệu từ host (CPU) vào device (GPU)
    cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_gradient, output_gradient, output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Gọi kernel backward để tính gradient đối với trọng số và độ chệch
    int blocks = (output_size + 255) / 256;
    backward_kernel<<<blocks, 256>>>(d_input, d_output_gradient, d_weights, d_weight_gradients, d_bias_gradients, input_size, output_size);
    cudaDeviceSynchronize();

    // Sao chép gradient về host
    cudaMemcpy(weight_gradients, d_weight_gradients, input_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(bias_gradients, d_bias_gradients, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Giải phóng bộ nhớ trên device
    cudaFree(d_input);
    cudaFree(d_output_gradient);
    cudaFree(d_weights);
    cudaFree(d_weight_gradients);
    cudaFree(d_bias_gradients);
}

// Phương thức cập nhật trọng số
void DenseLayer::update_weights(float* weight_gradients, float* bias_gradients, float learning_rate, int batch_size) {
    float *d_weights, *d_weight_gradients, *d_biases, *d_bias_gradients;

    // Cấp phát bộ nhớ trên device (GPU)
    cudaMalloc(&d_weights, input_size * output_size * sizeof(float));
    cudaMalloc(&d_weight_gradients, input_size * output_size * sizeof(float));
    cudaMalloc(&d_biases, output_size * sizeof(float));
    cudaMalloc(&d_bias_gradients, output_size * sizeof(float));

    // Sao chép dữ liệu từ host (CPU) vào device (GPU)
    cudaMemcpy(d_weights, weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_gradients, weight_gradients, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_gradients, bias_gradients, output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Gọi kernel update_weights để cập nhật trọng số và độ chệch
    int blocks = (output_size + 255) / 256;
    update_weights_kernel<<<blocks, 256>>>(d_weights, d_weight_gradients, d_biases, d_bias_gradients, learning_rate, input_size, output_size);
    cudaDeviceSynchronize();

    // Sao chép trọng số và độ chệch đã cập nhật về host
    cudaMemcpy(weights, d_weights, input_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(biases, d_biases, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Giải phóng bộ nhớ trên device
    cudaFree(d_weights);
    cudaFree(d_weight_gradients);
    cudaFree(d_biases);
    cudaFree(d_bias_gradients);
}

// Destructor
DenseLayer::~DenseLayer() {
    delete[] weights;
    delete[] biases;
}
