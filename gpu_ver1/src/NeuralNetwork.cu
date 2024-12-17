#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size)
    : layer1(input_size, hidden_size, new ReLU()),
      layer2(hidden_size, hidden_size, new ReLU()),
      layer3(hidden_size, output_size, new Softmax()) {}

std::vector<float> NeuralNetwork::forward(const std::vector<float>& input) {
    std::vector<float> x = layer1.forward(input);
    x = layer2.forward(x);
    x = layer3.forward(x);
    return x;
}

void NeuralNetwork::backward(const std::vector<float>& input, const std::vector<float>& target) {
    // Backpropagation logic
}

void NeuralNetwork::update_weights(float learning_rate) {
    // Cập nhật trọng số với gradient descent
}

NeuralNetwork::~NeuralNetwork() {
    // Xử lý giải phóng bộ nhớ GPU nếu cần
}
