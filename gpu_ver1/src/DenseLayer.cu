#include "DenseLayer.h"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include "Kernel.h"
#include <random>
#include <cmath>
#include "Macro.h"

DenseLayer::DenseLayer(int input_size, int output_size, ActivationFunction* activation)
    : input_size(input_size), output_size(output_size), activation(activation)
{
    weights = new float[input_size * output_size];
    biases = new float[output_size];

    weight_gradients = new float[input_size * output_size];
    bias_gradients = new float[output_size];

    int max_batch = 1024;
    last_input = new float[input_size * max_batch];
    last_output = new float[output_size * max_batch];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, sqrtf(2.0f / input_size));

    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            weights[i * input_size + j] = dist(gen);
        }
        biases[i] = 0.0f;
    }
}

// Forward batch
void DenseLayer::forward(float* input, float* output, int batch_size) {
    memcpy(last_input, input, input_size * batch_size * sizeof(float));

    float *d_input, *d_output, *d_weights, *d_biases;
    CHECK(cudaMalloc(&d_input, input_size * batch_size * sizeof(float)));
    CHECK(cudaMalloc(&d_output, output_size * batch_size * sizeof(float)));
    CHECK(cudaMalloc(&d_weights, input_size * output_size * sizeof(float)));
    CHECK(cudaMalloc(&d_biases, output_size * sizeof(float)));

    CHECK(cudaMemcpy(d_input, input, input_size * batch_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_weights, weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_biases, biases, output_size * sizeof(float), cudaMemcpyHostToDevice));

    int total_threads = output_size * batch_size;
    int blocks = (total_threads + 255) / 256;
    forward_kernel<<<blocks, 256>>>(d_input, d_output, d_weights, d_biases, input_size, output_size, batch_size);
    CHECK(cudaDeviceSynchronize());

    float* linear_output = new float[output_size * batch_size];
    CHECK(cudaMemcpy(linear_output, d_output, output_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost));

    for (int b = 0; b < batch_size; b++) {
        activation->activate(&linear_output[b * output_size], &output[b * output_size], output_size);
    }

    memcpy(last_output, output, output_size * batch_size * sizeof(float));

    delete[] linear_output;
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
    CHECK(cudaFree(d_weights));
    CHECK(cudaFree(d_biases));
}

// Backward batch
void DenseLayer::backward(float* output_gradient, float* input_gradient, int batch_size) {
    float* d_act = new float[output_size * batch_size];
    memcpy(d_act, output_gradient, output_size * batch_size * sizeof(float));

    for (int b = 0; b < batch_size; b++) {
        activation->derivative(&last_output[b * output_size], &d_act[b * output_size], output_size);
    }

    float *d_input, *d_act_dev, *d_weights, *d_wgrad, *d_bgrad, *d_igrad;
    CHECK(cudaMalloc(&d_input, input_size * batch_size * sizeof(float)));
    CHECK(cudaMalloc(&d_act_dev, output_size * batch_size * sizeof(float)));
    CHECK(cudaMalloc(&d_weights, input_size * output_size * sizeof(float)));
    CHECK(cudaMalloc(&d_wgrad, input_size * output_size * sizeof(float)));
    CHECK(cudaMalloc(&d_bgrad, output_size * sizeof(float)));
    CHECK(cudaMalloc(&d_igrad, input_size * batch_size * sizeof(float)));

    CHECK(cudaMemcpy(d_input, last_input, input_size * batch_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_act_dev, d_act, output_size * batch_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_weights, weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice));

    CHECK(cudaMemset(d_wgrad, 0, input_size * output_size * sizeof(float)));
    CHECK(cudaMemset(d_bgrad, 0, output_size * sizeof(float)));
    CHECK(cudaMemset(d_igrad, 0, input_size * batch_size * sizeof(float)));

    int total_threads = output_size * batch_size;
    int blocks = (total_threads + 255) / 256;
    backward_kernel<<<blocks, 256>>>(d_input, d_act_dev, d_weights, d_wgrad, d_bgrad, d_igrad, input_size, output_size, batch_size);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(weight_gradients, d_wgrad, input_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(bias_gradients, d_bgrad, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    if (input_gradient) {
        CHECK(cudaMemcpy(input_gradient, d_igrad, input_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost));
    }

    delete[] d_act;

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_act_dev));
    CHECK(cudaFree(d_weights));
    CHECK(cudaFree(d_wgrad));
    CHECK(cudaFree(d_bgrad));
    CHECK(cudaFree(d_igrad));
}

// Update weights
void DenseLayer::update_weights(float learning_rate, int batch_size) {
    float lr = learning_rate / (float)batch_size;

    float *d_weights, *d_weight_gradients, *d_biases, *d_bias_gradients;
    CHECK(cudaMalloc(&d_weights, input_size * output_size * sizeof(float)));
    CHECK(cudaMalloc(&d_weight_gradients, input_size * output_size * sizeof(float)));
    CHECK(cudaMalloc(&d_biases, output_size * sizeof(float)));
    CHECK(cudaMalloc(&d_bias_gradients, output_size * sizeof(float)));

    CHECK(cudaMemcpy(d_weights, weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_weight_gradients, weight_gradients, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_biases, biases, output_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias_gradients, bias_gradients, output_size * sizeof(float), cudaMemcpyHostToDevice));

    int blocks = (output_size + 255) / 256;
    update_weights_kernel<<<blocks, 256>>>(d_weights, d_weight_gradients, d_biases, d_bias_gradients, lr, input_size, output_size);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(weights, d_weights, input_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(biases, d_biases, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_weights));
    CHECK(cudaFree(d_weight_gradients));
    CHECK(cudaFree(d_biases));
    CHECK(cudaFree(d_bias_gradients));
}

DenseLayer::~DenseLayer() {
    delete[] weights;
    delete[] biases;
    delete[] weight_gradients;
    delete[] bias_gradients;
    delete[] last_input;
    delete[] last_output;
    delete activation;
}
