#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "activation_function.h"
#include "DenseLayer.h"

class NeuralNetwork {
public:
    DenseLayer layer1;
    DenseLayer layer2;
    DenseLayer layer3;

    NeuralNetwork(int input_size, int hidden_size, int output_size);

    // Forward pass
    std::vector<float> forward(const std::vector<float>& input);

    // Backward pass
    void backward(const std::vector<float>& input, const std::vector<float>& target);

    // Cập nhật trọng số
    void update_weights(float learning_rate);

    // Giải phóng bộ nhớ GPU
    ~NeuralNetwork();
};

#endif
