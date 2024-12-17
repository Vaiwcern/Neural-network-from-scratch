#include "ANN.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

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

void ANN::train(float* train_input, float* train_output, int batch_size, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Giả sử train_input và train_output đã được chia thành các batch
        for (int i = 0; i < batch_size; ++i) {
            // Forward pass
            float* output = new float[layer3->output_size];
            forward(&train_input[i * layer1->input_size], output);

            // Tính toán gradient (có thể sử dụng mất mát như CrossEntropy)
            float* output_gradient = new float[layer3->output_size];
            // Tính toán gradient ở lớp đầu ra
            // output_gradient = softmax_loss(output, train_output[i]);

            // Backward pass
            backward(train_input, output_gradient, batch_size);

            // Cập nhật trọng số và độ chệch
            update_weights(output_gradient, output_gradient, batch_size);

            delete[] output;
            delete[] output_gradient;
        }
        std::cout << "Epoch " << epoch + 1 << " completed!" << std::endl;
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
