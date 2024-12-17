// dense_layer.h
#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <vector>

class DenseLayer {
public:
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    int input_size, output_size;
    ActivationFunction* activation; // Pointer to the activation function

    DenseLayer(int input_size, int output_size, ActivationFunction* activation);
    std::vector<float> forward(const std::vector<float>& input); // Apply activation after forward pass
};

#endif
