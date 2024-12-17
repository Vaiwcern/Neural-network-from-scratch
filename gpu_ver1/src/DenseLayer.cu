// dense_layer.cpp
#include "DenseLayer.h"
#include <cstdlib>
#include <ctime>

DenseLayer::DenseLayer(int input_size, int output_size, ActivationFunction* activation)
    : input_size(input_size), output_size(output_size), activation(activation) 
{
    weights.resize(output_size, std::vector<float>(input_size));
    biases.resize(output_size);

    srand(time(0));
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            weights[i][j] = ((float)rand() / RAND_MAX) * 2 - 1; 
        }
        biases[i] = ((float)rand() / RAND_MAX) * 2 - 1; 
    }
}

std::vector<float> DenseLayer::forward(const std::vector<float>& input) {
    
}
