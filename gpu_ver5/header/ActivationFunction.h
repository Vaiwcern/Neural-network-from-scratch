#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <vector>
#include <cuda_fp16.h>  // Thêm thư viện để làm việc với float16

// Lớp cơ sở ActivationFunction
class ActivationFunction {
public:
    virtual void activate(half* input, half* output, int size) const = 0;  // Hàm kích hoạt cho toàn bộ vector
    virtual void derivative(half* output, half* d_output, int size) const = 0; // Đạo hàm hàm kích hoạt
    virtual ~ActivationFunction() {}
};

// Lớp con ReLU
class ReLU : public ActivationFunction {
public:
    void activate(half* input, half* output, int size) const override;  // Kích hoạt ReLU cho toàn bộ vector
    void derivative(half* output, half* d_output, int size) const override; // Đạo hàm ReLU
};

// Lớp con Softmax
// Lưu ý: Softmax + CrossEntropy loss có đạo hàm thuận tiện (d_output = output - target),
// nên với Softmax layer cuối (dùng CrossEntropy), ta có thể bỏ qua việc tính đạo hàm activation vì đã gộp vào loss.
class Softmax : public ActivationFunction {
public:
    void activate(half* input, half* output, int size) const override;  // Softmax cho toàn bộ vector
    void derivative(half* output, half* d_output, int size) const override; // Không thực sự cần cho output layer (CE loss)
};

#endif
