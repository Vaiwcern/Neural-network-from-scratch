#include "ANN.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iterator>
#include "CudaHelper.h"

ANN::ANN(int input_size, int hidden_size, int output_size, float learning_rate) {
    // Khởi tạo các lớp DenseLayer với các lớp kích hoạt tương ứng
    layer1 = new DenseLayer(input_size, hidden_size, new ReLU());  // Lớp ẩn 1 với ReLU
    layer2 = new DenseLayer(hidden_size, hidden_size, new ReLU());  // Lớp ẩn 2 với ReLU
    layer3 = new DenseLayer(hidden_size, output_size, new Softmax());  // Lớp đầu ra với Softmax

    this->learning_rate = learning_rate;
}

void ANN::forward(float* input, float* output) {
    // Lớp 1: Forward pass qua lớp đầu tiên
    float* hidden1_output = new float[layer1->output_size];
    layer1->forward(input, hidden1_output);

    // Lớp 2: Forward pass qua lớp ẩn thứ 2
    float* hidden2_output = new float[layer2->output_size];
    layer2->forward(hidden1_output, hidden2_output);

    // Lớp 3: Forward pass qua lớp đầu ra (Softmax)
    layer3->forward(hidden2_output, output);

    delete[] hidden1_output;
    delete[] hidden2_output;
}

void ANN::backward(float* input, float* output_gradient, int batch_size) {
    // Lớp 3: Backward pass qua lớp đầu ra
    float* weight_gradients3 = new float[layer3->input_size * layer3->output_size];
    float* bias_gradients3 = new float[layer3->output_size];
    layer3->backward(input, output_gradient, weight_gradients3, bias_gradients3, batch_size);

    // Lớp 2: Backward pass qua lớp ẩn thứ 2
    float* weight_gradients2 = new float[layer2->input_size * layer2->output_size];
    float* bias_gradients2 = new float[layer2->output_size];
    layer2->backward(input, output_gradient, weight_gradients2, bias_gradients2, batch_size);

    // Lớp 1: Backward pass qua lớp ẩn thứ 1
    float* weight_gradients1 = new float[layer1->input_size * layer1->output_size];
    float* bias_gradients1 = new float[layer1->output_size];
    layer1->backward(input, output_gradient, weight_gradients1, bias_gradients1, batch_size);

    delete[] weight_gradients3;
    delete[] bias_gradients3;
    delete[] weight_gradients2;
    delete[] bias_gradients2;
    delete[] weight_gradients1;
    delete[] bias_gradients1;
}

void ANN::update_weights(float* weight_gradients, float* bias_gradients, int batch_size) {
    // Cập nhật trọng số và độ chệch cho từng lớp
    layer1->update_weights(weight_gradients, bias_gradients, learning_rate, batch_size);
    layer2->update_weights(weight_gradients, bias_gradients, learning_rate, batch_size);
    layer3->update_weights(weight_gradients, bias_gradients, learning_rate, batch_size);
}

void ANN::train(float* train_input, float* train_output, int num_samples, int batch_size, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < num_samples; i += batch_size) {
            // Forward pass
            float* output = new float[layer3->output_size];
            forward(&train_input[i * layer1->input_size], output);

            // Tính toán Cross-Entropy loss gradient song song cho tất cả các phần tử trong batch
            float* d_loss;
            CHECK(cudaMalloc(&d_loss, sizeof(float) * layer3->output_size));
            CHECK(cudaMemset(d_loss, 0, sizeof(float) * layer3->output_size));
    
            // cross_entropy_loss_gradient_kernel<<<(layer3->output_size + 255) / 256, 256>>>(
            //     output, &train_output[i * layer3->output_size], d_loss, layer3->output_size
            // );
            // CHECK(cudaDeviceSynchronize());  // Đồng bộ hóa để đảm bảo kernel đã hoàn thành

            // float* gradient = new float[layer3->output_size];
            // CHECK(cudaMemcpy(gradient, d_loss, sizeof(float) * layer3->output_size, cudaMemcpyDeviceToHost));

            // Backward pass
            // backward(train_input, gradient, batch_size);

            // Cập nhật trọng số và độ chệch
            // update_weights(gradient, gradient, batch_size);

            delete[] output;
            delete[] gradient;
            CHECK(cudaFree(d_loss));  // Giải phóng bộ nhớ GPU

            break;
        }
        std::cout << "Epoch " << epoch + 1 << " completed!" << std::endl;

        break;
    }
}


void ANN::eval(float* test_input, float* test_output, int test_size) {
    int correct_predictions = 0;
    for (int i = 0; i < test_size; ++i) {
        // Dự đoán cho đầu vào
        float* output = new float[layer3->output_size];
        forward(&test_input[i * layer1->input_size], output);

        // Chuyển output thành vector để sử dụng std::max_element và std::distance
        std::vector<float> output_vector(output, output + layer3->output_size);

        // Tính toán nhãn dự đoán
        int predicted_label = std::distance(output_vector.begin(), std::max_element(output_vector.begin(), output_vector.end()));
        int actual_label = test_output[i];

        if (predicted_label == actual_label) {
            correct_predictions++;
        }

        delete[] output;
    }

    float accuracy = (float)correct_predictions / test_size;
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
}




ANN::~ANN() {
    delete layer1;
    delete layer2;
    delete layer3;
}
