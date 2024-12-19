// DenseLayer.h
#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "ActivationFunction.h"
#include <cuda_runtime.h>

class DenseLayer {
public:
    int input_size;
    int output_size;
    ActivationFunction* activation;

    float *weights;         // host
    float *biases;          // host
    float *weight_gradients; // host
    float *bias_gradients;   // host

    float *last_input;   // host
    float *last_output;  // host

    // GPU buffers (device)
    // allocate once and reuse
    float *d_input;
    float *d_output;
    float *d_weights;
    float *d_biases;
    float *d_linear_output;
    float *d_wgrad;
    float *d_bgrad;
    float *d_igrad; 
    float *d_act;

    // max_batch được truyền vào hoặc định nghĩa sẵn
    int max_batch;

    DenseLayer(int input_size, int output_size, ActivationFunction* activation, int max_batch=1024);
    ~DenseLayer();

    void forward(float* input, float* output, int batch_size);
    void backward(float* output_gradient, float* input_gradient, int batch_size);
    void update_weights(float learning_rate, int batch_size, cudaStream_t stream);
};

#endif
