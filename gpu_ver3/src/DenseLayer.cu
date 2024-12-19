#include "DenseLayer.h"
#include "Kernel.h"
#include <random>
#include <cmath>
#include <cstring>
#include "Macro.h"
#include <cuda_fp16.h>  // Thư viện hỗ trợ kiểu dữ liệu half

DenseLayer::DenseLayer(int input_size, int output_size, ActivationFunction *activation, int max_batch)
    : input_size(input_size), output_size(output_size), activation(activation), max_batch(max_batch)
{
    // Chuyển đổi toàn bộ các biến sang kiểu half
    weights = new half[input_size * output_size];
    biases = new half[output_size];

    weight_gradients = new half[input_size * output_size];
    bias_gradients = new half[output_size];

    last_input = new half[input_size * max_batch];
    last_output = new half[output_size * max_batch];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, sqrtf(2.0f / input_size));

    for (int i = 0; i < output_size; ++i)
    {
        for (int j = 0; j < input_size; ++j)
        {
            weights[i * input_size + j] = __float2half(dist(gen)); // Chuyển đổi từ float sang half
        }
        biases[i] = __float2half(0.0f); // Chuyển đổi từ float sang half
    }

    // Allocate GPU memory once
    CHECK(cudaMalloc(&d_input, input_size * max_batch * sizeof(half)));
    CHECK(cudaMalloc(&d_output, output_size * max_batch * sizeof(half)));
    CHECK(cudaMalloc(&d_weights, input_size * output_size * sizeof(half)));
    CHECK(cudaMalloc(&d_biases, output_size * sizeof(half)));
    CHECK(cudaMalloc(&d_linear_output, output_size * max_batch * sizeof(half)));
    CHECK(cudaMalloc(&d_wgrad, input_size * output_size * sizeof(half)));
    CHECK(cudaMalloc(&d_bgrad, output_size * sizeof(half)));
    CHECK(cudaMalloc(&d_igrad, input_size * max_batch * sizeof(half)));
    CHECK(cudaMalloc(&d_act, output_size * max_batch * sizeof(half)));

    // Copy initial weights, biases to device
    CHECK(cudaMemcpy(d_weights, weights, input_size * output_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_biases, biases, output_size * sizeof(half), cudaMemcpyHostToDevice));
}

// Forward batch
void DenseLayer::forward(half *input, half *output, int batch_size)
{
    memcpy(last_input, input, input_size * batch_size * sizeof(half));

    // Copy input to device once
    CHECK(cudaMemcpy(d_input, input, input_size * batch_size * sizeof(half), cudaMemcpyHostToDevice));

    int threads = 512; // đảm bảo threads >= output_size
    int blocks = batch_size;
    size_t shared_mem_size = threads * sizeof(half); // vì ta dùng s_input[threads]
    forward_kernel<<<blocks, threads, shared_mem_size>>>(d_input, d_linear_output, d_weights, d_biases,
                                                         input_size, output_size, batch_size);
    CHECK(cudaDeviceSynchronize());

    // linear_output đã ở d_linear_output, apply activation trên host hoặc device
    // Ở đây ta làm activation trên host->device->host không hiệu quả,
    // ta nên implement activation kernel và gọi trực tiếp trên device.
    // Giả sử activation->activate(...) đã sử dụng kernel như ReLU hay softmax:
    // Ta gọi thẳng activation kernel trên d_linear_output -> d_output

    if (dynamic_cast<ReLU *>(activation))
    {
        int size = output_size * batch_size;
        int threads_act = 256;
        int blocks_act = (size + threads_act - 1) / threads_act;
        relu_kernel<<<blocks_act, threads_act>>>(d_linear_output, d_output, size);
        CHECK(cudaDeviceSynchronize());
    }
    else if (dynamic_cast<Softmax *>(activation))
    {
        // Softmax trên batch_size mẫu: Mỗi mẫu là 1 vector.
        for (int b = 0; b < batch_size; b++)
        {
            softmax_kernel<<<1, output_size, 2 * output_size * sizeof(half)>>>(d_linear_output + b * output_size, d_output + b * output_size, output_size);
        }
        CHECK(cudaDeviceSynchronize());
    }

    // Copy output về host
    CHECK(cudaMemcpy(output, d_output, output_size * batch_size * sizeof(half), cudaMemcpyDeviceToHost));
    memcpy(last_output, output, output_size * batch_size * sizeof(half));
}

// Backward batch
void DenseLayer::backward(half *output_gradient, half *input_gradient, int batch_size)
{
    // d_act = output_gradient trên device
    CHECK(cudaMemcpy(d_act, output_gradient, output_size * batch_size * sizeof(half), cudaMemcpyHostToDevice));

    // Gọi derivative kernel:
    int size = output_size * batch_size;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    // Chúng ta đã có last_output trên host, cần copy lên device để derivative.
    CHECK(cudaMemcpy(d_output, last_output, output_size * batch_size * sizeof(half), cudaMemcpyHostToDevice));
    relu_derivative_kernel<<<blocks, threads>>>(d_output, d_act, size);
    CHECK(cudaDeviceSynchronize());

    // bây giờ d_act là output_gradient sau activation derivative
    CHECK(cudaMemcpy(d_input, last_input, input_size * batch_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_wgrad, 0, input_size * output_size * sizeof(half)));
    CHECK(cudaMemset(d_bgrad, 0, output_size * sizeof(half)));
    CHECK(cudaMemset(d_igrad, 0, input_size * batch_size * sizeof(half)));

    int backward_blocks = output_size;
    int backward_threads = input_size;
    size_t shared_mem_size = input_size * sizeof(half); // Shared memory for weights

    backward_kernel<<<backward_blocks, backward_threads, shared_mem_size>>>( 
        d_input,
        d_act,
        d_weights,
        d_wgrad,
        d_bgrad,
        d_igrad,
        input_size,
        output_size,
        batch_size);
    CHECK(cudaDeviceSynchronize());

    // Copy gradients về host
    CHECK(cudaMemcpy(weight_gradients, d_wgrad, input_size * output_size * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(bias_gradients, d_bgrad, output_size * sizeof(half), cudaMemcpyDeviceToHost));

    if (input_gradient)
    {
        CHECK(cudaMemcpy(input_gradient, d_igrad, input_size * batch_size * sizeof(half), cudaMemcpyDeviceToHost));
    }
}

// Update weights
void DenseLayer::update_weights(half learning_rate, int batch_size, cudaStream_t stream)
{
    // Copy host gradients to GPU asynchronously in the given stream
    CHECK(cudaMemcpyAsync(d_wgrad, weight_gradients, input_size * output_size * sizeof(half), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(d_bgrad, bias_gradients, output_size * sizeof(half), cudaMemcpyHostToDevice, stream));

    int threads = 512;
    int blocks = (output_size + threads - 1) / threads;
    half lr = __float2half(learning_rate) / (half)batch_size;

    // Launch the kernel in the provided stream
    update_weights_kernel<<<blocks, threads, 0, stream>>>(d_weights, d_wgrad, d_biases, d_bgrad, lr, input_size, output_size);

    // Copy weights and biases back to host asynchronously (if needed)
    CHECK(cudaMemcpyAsync(weights, d_weights, input_size * output_size * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(biases, d_biases, output_size * sizeof(half), cudaMemcpyDeviceToHost, stream));
}

DenseLayer::~DenseLayer()
{
    delete[] weights;
    delete[] biases;
    delete[] weight_gradients;
    delete[] bias_gradients;
    delete[] last_input;
    delete[] last_output;
    delete activation;

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
    CHECK(cudaFree(d_weights));
    CHECK(cudaFree(d_biases));
    CHECK(cudaFree(d_linear_output));
    CHECK(cudaFree(d_wgrad));
    CHECK(cudaFree(d_bgrad));
    CHECK(cudaFree(d_igrad));
    CHECK(cudaFree(d_act));
}
