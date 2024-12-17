#include <iostream>
#include <vector>
#include "neural_network.h"
#include "kernel.h"

int main() {
    int input_size = 784;  // Ví dụ với dữ liệu MNIST
    int hidden_size = 128; // Số lượng nơ-ron trong lớp ẩn
    int output_size = 10;  // Số lớp đầu ra (10 lớp cho bài toán phân loại)

    NeuralNetwork nn(input_size, hidden_size, output_size);

    // Giả sử input_data và output_gradient được chuẩn bị từ trước
    std::vector<float> input_data(input_size);  // Cần cung cấp dữ liệu thực tế
    std::vector<float> output_gradient(hidden_size); // Cần tính toán gradient thực tế

    // Forward pass
    nn.forward(input_data);

    // Backward pass
    nn.backward(input_data, output_gradient);

    // Cập nhật trọng số
    nn.update_weights(0.01f);

    std::cout << "Training completed." << std::endl;

    return 0;
}
