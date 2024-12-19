#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "ActivationFunction.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // Thêm thư viện để làm việc với half precision

class DenseLayer {
public:
    int input_size;
    int output_size;
    ActivationFunction* activation;

    // Các biến host
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

    int max_batch;

    DenseLayer(int input_size, int output_size, ActivationFunction* activation, int max_batch=1024);
    ~DenseLayer();

    void forward(half* input, half* output, int batch_size);
    void backward(half* output_gradient, half* input_gradient, int batch_size);
    void update_weights(float learning_rate, int batch_size, cudaStream_t stream);
};

#endif
