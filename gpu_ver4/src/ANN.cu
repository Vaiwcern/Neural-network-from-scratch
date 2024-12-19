#include "ANN.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_fp16.h>  // Thêm thư viện để làm việc với half precision

// Constructor
ANN::ANN(int input_size, int hidden_size, int output_size, float learning_rate) {
    // Tạo các layer với activation tương ứng
    layer1 = new DenseLayer(input_size, hidden_size, new ReLU());
    layer2 = new DenseLayer(hidden_size, hidden_size, new ReLU());
    layer3 = new DenseLayer(hidden_size, output_size, new Softmax());

    this->learning_rate = learning_rate;
}

void ANN::forward(half* input, half* output, int batch_size) {
    // Forward qua các layers
    layer1->forward(input, layer1->last_output, batch_size);
    layer2->forward(layer1->last_output, layer2->last_output, batch_size);
    layer3->forward(layer2->last_output, layer3->last_output, batch_size);

    // Sao chép output
    memcpy(output, layer3->last_output, layer3->output_size * batch_size * sizeof(half));
}

void ANN::backward(half* input, half* target, int batch_size) {
    // Với Softmax + CrossEntropy: dOutput = output - target
    // Không chia batch_size ở đây, giống code CPU
    half* output_gradient = new half[layer3->output_size * batch_size];
    for (int i = 0; i < layer3->output_size * batch_size; i++) {
        output_gradient[i] = __float2half(__half2float(layer3->last_output[i]) - __half2float(target[i])); // Không chia batch_size
    }

    half* grad2 = new half[layer2->output_size * batch_size]; // gradient cho layer2 output
    half* grad1 = new half[layer1->output_size * batch_size]; // gradient cho layer1 output

    // Backward qua layer3
    layer3->backward(output_gradient, grad2, batch_size);

    // Backward qua layer2
    layer2->backward(grad2, grad1, batch_size);

    // Backward qua layer1
    layer1->backward(grad1, nullptr, batch_size); 

    delete[] output_gradient;
    delete[] grad2;
    delete[] grad1;
}

void ANN::update_weights(int batch_size) {
    // Tạo 3 stream cho các lớp
    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    // Gọi hàm update_weights cho từng lớp, mỗi lớp trong một stream riêng biệt
    layer1->update_weights(learning_rate, batch_size, stream1);
    layer2->update_weights(learning_rate, batch_size, stream2);
    layer3->update_weights(learning_rate, batch_size, stream3);

    // Đợi tất cả các stream hoàn thành
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);

    // Hủy các stream sau khi sử dụng
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
}

void ANN::train(half* train_input, unsigned char* train_labels, int num_samples, int epochs, int batch_size) {
    int steps_per_epoch = num_samples / batch_size;
    std::cout << "[TRAIN] Starting training..." << std::endl;
    std::cout << "[TRAIN] Num samples: " << num_samples << ", Batch size: " << batch_size 
              << ", Steps per epoch: " << steps_per_epoch << ", Epochs: " << epochs << std::endl;

    half* batch_output = new half[layer3->output_size * batch_size];
    half* batch_target = new half[layer3->output_size * batch_size];

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int correct_count = 0;

        std::cout << "[TRAIN] Epoch " << (epoch+1) << " started." << std::endl;

        for (int step = 0; step < steps_per_epoch; step++) {
            half* input_batch = &train_input[step * batch_size * layer1->input_size];
            unsigned char* label_batch = &train_labels[step * batch_size];

            // Forward
            forward(input_batch, batch_output, batch_size);

            // Tạo one-hot target
            for (int b = 0; b < batch_size; b++) {
                for (int i = 0; i < layer3->output_size; i++) {
                    batch_target[b * layer3->output_size + i] = __float2half(0.0f);
                }
                batch_target[b * layer3->output_size + label_batch[b]] = __float2half(1.0f);
            }

            // Compute loss
            float loss = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                float pred_for_label = __half2float(batch_output[b * layer3->output_size + label_batch[b]]);
                loss -= logf(pred_for_label + 1e-9f);
            }
            loss /= batch_size;
            epoch_loss += loss;

            // Compute accuracy
            for (int b = 0; b < batch_size; b++) {
                int pred = (int)(std::distance(&batch_output[b * layer3->output_size], 
                                               std::max_element(&batch_output[b * layer3->output_size],
                                                                &batch_output[b * layer3->output_size + layer3->output_size])));
                if (pred == (int)label_batch[b]) correct_count++;
            }

            if (step % 100 == 0) {
                std::cout << "[TRAIN][Epoch " << (epoch+1) << "][Step " << step << "] Loss: " << loss << std::endl;
            }

            // Backward
            backward(input_batch, batch_target, batch_size);

            // Update weights
            update_weights(batch_size);
        }

        float avg_loss = epoch_loss / steps_per_epoch;
        float accuracy = (float)correct_count / (steps_per_epoch * batch_size) * 100.0f;
        std::cout << "[TRAIN] Epoch " << (epoch+1) << " completed. Avg Loss = " << avg_loss 
                  << ", Accuracy = " << accuracy << "%" << std::endl;
    }

    delete[] batch_output;
    delete[] batch_target;
    std::cout << "[TRAIN] Training finished." << std::endl;
}

void ANN::eval(half* test_input, unsigned char* test_labels, int test_size) {
    int correct_predictions = 0;
    for (int i = 0; i < test_size; i++) {
        half* output = new half[layer3->output_size];
        forward(&test_input[i * layer1->input_size], output, 1);

        int predicted_label = (int)std::distance(output, std::max_element(output, output + layer3->output_size));
        int actual_label = (int)test_labels[i];

        if (predicted_label == actual_label) {
            correct_predictions++;
        }

        delete[] output;
    }

    float accuracy = (float)correct_predictions / test_size;
    std::cout << "Test Accuracy: " << accuracy * 100 << "%" << std::endl;
}

ANN::~ANN() {
    delete layer1;
    delete layer2;
    delete layer3;
}
