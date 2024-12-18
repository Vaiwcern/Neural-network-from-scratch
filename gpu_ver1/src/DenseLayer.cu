#include "DenseLayer.h"
#include "CudaHelper.h"
#include <cstdlib>
#include <ctime>
#include <iostream>

using namespace std;

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
    cout << "FORWARD LẦN 1:" << endl;
    cout << "Input size: " << input_size << ", Output size: " << output_size << endl;
    cout << "Weights matrix (input_size x output_size): " << input_size << " x " << output_size << endl;

    // Print input values for debugging
    cout << "Input: ";
    for (int i = 0; i < input_size; ++i) {
        cout << input[i] << " ";
    }
    cout << endl;

    // Print weights matrix (for debugging)
    cout << "Weights (Matrix):" << endl;
    for (int i = 0; i < output_size; ++i) {  // Loop through rows (output_size)
        for (int j = 0; j < input_size; ++j) {  // Loop through columns (input_size)
            cout << weights[i * input_size + j] << " ";  // Print element at [i, j]
        }
        cout << endl;  // New line after each row
    }

    // Memory allocation and CUDA code for forward pass
    float *d_input, *d_output, *d_weights, *d_biases;

    CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
    CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));
    CHECK(cudaMalloc(&d_weights, input_size * output_size * sizeof(float)));
    CHECK(cudaMalloc(&d_biases, output_size * sizeof(float)));

    // Copy data to device
    CHECK(cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_weights, weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_biases, biases, output_size * sizeof(float), cudaMemcpyHostToDevice));

    // Perform the forward pass with kernel
    int blocks = (output_size + 255) / 256;
    forward_kernel<<<blocks, 256>>>(d_input, d_output, d_weights, d_biases, input_size, output_size);
    CHECK(cudaGetLastError());  // Check for kernel launch errors
    CHECK(cudaDeviceSynchronize());  // Ensure kernel execution is finished

    cout << "-------HIHI---------" << "\n";

    // Apply activation function (ReLU or Softmax)
    activation->activate(d_output, d_output, output_size);

    cout << "-------HIHI---------" << "\n";

    // Copy result back to host
    CHECK(cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Print output for debugging
    cout << "Output: ";
    for (int i = 0; i < output_size; ++i) {
        cout << output[i] << " ";
    }
    cout << endl;

    // Free memory
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
    CHECK(cudaFree(d_weights));
    CHECK(cudaFree(d_biases));
}


// Phương thức backward pass (tính toán gradient)
void DenseLayer::backward(float* input, float* output_gradient, float* weight_gradients, float* bias_gradients, int batch_size) {
    float *d_input, *d_output_gradient, *d_weights, *d_weight_gradients, *d_bias_gradients;

    // Cấp phát bộ nhớ trên device (GPU)
    CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
    CHECK(cudaMalloc(&d_output_gradient, output_size * sizeof(float)));
    CHECK(cudaMalloc(&d_weights, input_size * output_size * sizeof(float)));
    CHECK(cudaMalloc(&d_weight_gradients, input_size * output_size * sizeof(float)));
    CHECK(cudaMalloc(&d_bias_gradients, output_size * sizeof(float)));

    // Sao chép dữ liệu từ host (CPU) vào device (GPU)
    CHECK(cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_output_gradient, output_gradient, output_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_weights, weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice));

    // Gọi kernel backward để tính gradient đối với trọng số và độ chệch
    int blocks = (output_size + 255) / 256;
    backward_kernel<<<blocks, 256>>>(d_input, d_output_gradient, d_weights, d_weight_gradients, d_bias_gradients, input_size, output_size);
    CHECK(cudaDeviceSynchronize());

    // Sao chép gradient về host
    CHECK(cudaMemcpy(weight_gradients, d_weight_gradients, input_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(bias_gradients, d_bias_gradients, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Giải phóng bộ nhớ trên device
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output_gradient));
    CHECK(cudaFree(d_weights));
    CHECK(cudaFree(d_weight_gradients));
    CHECK(cudaFree(d_bias_gradients));
}

// Phương thức cập nhật trọng số
void DenseLayer::update_weights(float* weight_gradients, float* bias_gradients, float learning_rate, int batch_size) {
    float *d_weights, *d_weight_gradients, *d_biases, *d_bias_gradients;

    // Cấp phát bộ nhớ trên device (GPU)
    CHECK(cudaMalloc(&d_weights, input_size * output_size * sizeof(float)));
    CHECK(cudaMalloc(&d_weight_gradients, input_size * output_size * sizeof(float)));
    CHECK(cudaMalloc(&d_biases, output_size * sizeof(float)));
    CHECK(cudaMalloc(&d_bias_gradients, output_size * sizeof(float)));

    // Sao chép dữ liệu từ host (CPU) vào device (GPU)
    CHECK(cudaMemcpy(d_weights, weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_weight_gradients, weight_gradients, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_biases, biases, output_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias_gradients, bias_gradients, output_size * sizeof(float), cudaMemcpyHostToDevice));

    // Gọi kernel update_weights để cập nhật trọng số và độ chệch
    int blocks = (output_size + 255) / 256;
    update_weights_kernel<<<blocks, 256>>>(d_weights, d_weight_gradients, d_biases, d_bias_gradients, learning_rate, input_size, output_size);
    CHECK(cudaDeviceSynchronize());

    // Sao chép trọng số và độ chệch đã cập nhật về host
    CHECK(cudaMemcpy(weights, d_weights, input_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(biases, d_biases, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Giải phóng bộ nhớ trên device
    CHECK(cudaFree(d_weights));
    CHECK(cudaFree(d_weight_gradients));
    CHECK(cudaFree(d_biases));
    CHECK(cudaFree(d_bias_gradients));
}

// Destructor
DenseLayer::~DenseLayer() {
    delete[] weights;
    delete[] biases;
}
