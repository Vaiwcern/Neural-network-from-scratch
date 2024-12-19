#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "ActivationFunction.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // Thư viện hỗ trợ kiểu dữ liệu half

class DenseLayer {
public:
    int input_size;
    int output_size;
    ActivationFunction* activation;

    // Thay float bằng half
    half *weights;         // host
    half *biases;          // host
    half *weight_gradients; // host
    half *bias_gradients;   // host

    half *last_input;   // host
    half *last_output;  // host

    // GPU buffers (device)
    // allocate once and reuse
    half *d_input;
    half *d_output;
    half *d_weights;
    half *d_biases;
    half *d_linear_output;
    half *d_wgrad;
    half *d_bgrad;
    half *d_igrad; 
    half *d_act;

    // max_batch được truyền vào hoặc định nghĩa sẵn
    int max_batch;

    // Constructor và Destructor
    DenseLayer(int input_size, int output_size, ActivationFunction* activation, int max_batch=1024);
    ~DenseLayer();

    // Phương thức forward, backward và update_weights sử dụng half
    void forward(half* input, half* output, int batch_size);
    void backward(half* output_gradient, half* input_gradient, int batch_size);
    void update_weights(half learning_rate, int batch_size, cudaStream_t stream);
};

#endif
