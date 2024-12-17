#ifndef ANN_H
#define ANN_H

#include <vector>
#include "DenseLayer.h"  // Để sử dụng lớp DenseLayer
#include "ActivateFunction.h"  // Để sử dụng các lớp ActivationFunction

class ANN {
public:
    DenseLayer* layer1;
    DenseLayer* layer2;
    DenseLayer* layer3;  // Lớp đầu ra có kích thước 10 (cho MNIST hoặc bài toán phân loại)

    float learning_rate;

    // Khởi tạo ANN với các lớp DenseLayer
    ANN(int input_size, int hidden_size, int output_size, float learning_rate);

    // Forward pass: Tính toán đầu ra của ANN
    void forward(float* input, float* output);

    // Backward pass: Tính toán gradient cho các lớp
    void backward(float* input, float* output_gradient, int batch_size);

    // Cập nhật trọng số và độ chệch
    void update_weights(float* weight_gradients, float* bias_gradients, int batch_size);

    // Phương thức huấn luyện ANN
    void train(float* train_input, float* train_output, int batch_size, int epochs);

    // Phương thức đánh giá mô hình (eval)
    void eval(float* test_input, float* test_output, int test_size);

    // Destructor để giải phóng bộ nhớ
    ~ANN();
};

#endif
