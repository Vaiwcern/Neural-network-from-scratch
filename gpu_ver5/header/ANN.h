#ifndef ANN_H
#define ANN_H

#include <vector>
#include "DenseLayer.h"  // Để sử dụng lớp DenseLayer
#include "ActivationFunction.h"  // Để sử dụng các lớp ActivationFunction
#include <cuda_fp16.h>  // Thêm thư viện để làm việc với float16

class ANN {
public:
    DenseLayer* layer1;
    DenseLayer* layer2;
    DenseLayer* layer3;  // Lớp đầu ra

    float learning_rate;

    // Khởi tạo ANN với các lớp DenseLayer
    ANN(int input_size, int hidden_size, int output_size, float learning_rate);

    // Forward pass: Tính toán đầu ra của ANN
    void forward(half* input, half* output, int batch_size);

    // Backward pass: Tính toán gradient cho các lớp (sử dụng cross-entropy loss)
    void backward(half* input, half* target, int batch_size);

    // Cập nhật trọng số và độ chệch
    void update_weights(int batch_size);

    // Phương thức huấn luyện ANN
    void train(half* train_input, unsigned char* train_labels, int num_samples, int epochs, int batch_size);

    // Phương thức đánh giá mô hình (eval)
    void eval(half* test_input, unsigned char* test_labels, int test_size);

    // Destructor để giải phóng bộ nhớ
    ~ANN();
};

#endif