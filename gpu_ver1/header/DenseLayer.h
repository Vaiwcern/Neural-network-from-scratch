#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "ActivationFunction.h"

class DenseLayer {
public:
    int input_size;
    int output_size;
    ActivationFunction* activation;

    float* weights;
    float* biases;

    float* weight_gradients;
    float* bias_gradients;

    float* last_input;  // Giờ sẽ lưu cả batch input
    float* last_output; // Lưu cả batch output

    DenseLayer(int input_size, int output_size, ActivationFunction* activation);

    void forward(float* input, float* output, int batch_size);
    void backward(float* output_gradient, float* input_gradient, int batch_size);

    void update_weights(float learning_rate, int batch_size);

    ~DenseLayer();
};

#endif
