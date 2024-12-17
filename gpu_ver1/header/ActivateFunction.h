#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <vector>
#include "kernel.h"  // Để sử dụng các kernel trong lớp activation

// Lớp cơ sở ActivationFunction
class ActivationFunction {
public:
    virtual void activate(float* input, float* output, int size) const = 0;  // Hàm kích hoạt cho toàn bộ vector
};

// Lớp con ReLU
class ReLU : public ActivationFunction {
public:
    void activate(float* input, float* output, int size) const override;  // Kích hoạt ReLU cho toàn bộ vector
};

// Lớp con Softmax
class Softmax : public ActivationFunction {
public:
    void activate(float* input, float* output, int size) const override;  // Softmax cho toàn bộ vector
};

#endif
