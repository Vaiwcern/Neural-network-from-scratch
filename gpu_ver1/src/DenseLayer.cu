#include "DenseLayer.h"
#include "CudaHelper.h"
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
    cout << "FORWARD LẦN 1: "

    // In input, weights, biases, và output trước khi vào kernel (trên host)
    cout << "Input: ";
    for (int i = 0; i < input_size; ++i) {
        cout << input[i] << " ";
    }
    cout << endl;

    // In weights dưới dạng ma trận (input_size x output_size)
    cout << "Weights (Matrix):" << endl;
    for (int i = 0; i < output_size; ++i) {  // Duyệt qua các hàng (output_size)
        for (int j = 0; j < input_size; ++j) {  // Duyệt qua các cột (input_size)
            cout << weights[i * input_size + j] << " ";  // In phần tử [i, j] trong ma trận
        }
        cout << endl;  // Xuống dòng sau mỗi hàng
    }

    cout << "Biases: ";
    for (int i = 0; i < output_size; ++i) {
        cout << biases[i] << " ";
    }
    cout << endl;


    float *d_input, *d_output, *d_weights, *d_biases;

    // Cấp phát bộ nhớ trên device (GPU)
    CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
    CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));
    CHECK(cudaMalloc(&d_weights, input_size * output_size * sizeof(float)));
    CHECK(cudaMalloc(&d_biases, output_size * sizeof(float)));

    // Sao chép dữ liệu từ host (CPU) vào device (GPU)
    CHECK(cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_weights, weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_biases, biases, output_size * sizeof(float), cudaMemcpyHostToDevice));

    // Tính toán forward pass (tính tổng có trọng số và độ chệch)
    int blocks = (output_size + 255) / 256;
    forward_kernel<<<blocks, 256>>>(d_input, d_output, d_weights, d_biases, input_size, output_size);
    CHECK(cudaDeviceSynchronize());  // Đồng bộ hóa để đảm bảo kernel đã hoàn thành

    // Áp dụng hàm kích hoạt (ReLU hoặc Softmax)
    activation->activate(d_output, d_output, output_size);

    // Sao chép kết quả từ device về host
    CHECK(cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    // In kết quả output sau khi tính toán
    cout << "Output: ";
    for (int i = 0; i < output_size; ++i) {
        cout << output[i] << " ";
    }
    cout << endl;

    // Giải phóng bộ nhớ trên device
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
