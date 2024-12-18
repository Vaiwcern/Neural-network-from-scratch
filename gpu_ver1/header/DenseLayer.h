#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <vector>
#include "ActivateFunction.h"  // Để sử dụng các lớp kích hoạt
#include "Kernel.h"  // Để sử dụng các kernel CUDA

class DenseLayer {
public:
    int input_size, output_size;
    float *weights, *biases;
    ActivationFunction *activation;  // Con trỏ đến hàm kích hoạt (ReLU hoặc Softmax)

    DenseLayer(int input_size, int output_size, ActivationFunction* activation);
    
    // Phương thức forward pass
    void forward(float* input, float* output);

    // Phương thức backward pass (tính toán gradient)
    void backward(float* input, float* output_gradient, float* weight_gradients, float* bias_gradients, int batch_size);

    // Phương thức cập nhật trọng số
    void update_weights(float* weight_gradients, float* bias_gradients, float learning_rate, int batch_size);

    // Giải phóng bộ nhớ GPU
    ~DenseLayer();
};

#endif
