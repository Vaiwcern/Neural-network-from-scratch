#include "ANN.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cmath>

// Constructor
ANN::ANN(int input_size, int hidden_size, int output_size, float learning_rate) {
    // Khởi tạo các lớp DenseLayer với các lớp kích hoạt tương ứng
    layer1 = new DenseLayer(input_size, hidden_size, new ReLU());
    layer2 = new DenseLayer(hidden_size, hidden_size, new ReLU());
    layer3 = new DenseLayer(hidden_size, output_size, new Softmax());

    this->learning_rate = learning_rate;
}

void ANN::forward(float* input, float* output, int batch_size) {
    // Forward qua các lớp
    layer1->forward(input, layer1->last_output, batch_size);
    layer2->forward(layer1->last_output, layer2->last_output, batch_size);
    layer3->forward(layer2->last_output, layer3->last_output, batch_size);

    // Sao chép output cuối cùng ra ngoài (batch_size * output_size)
    memcpy(output, layer3->last_output, layer3->output_size * batch_size * sizeof(float));
}

void ANN::backward(float* input, float* target, int batch_size) {
    // Với Softmax + CrossEntropy: output_gradient = (output - target) / batch_size
    // output: layer3->last_output: (batch_size * output_size)
    // target: (batch_size * output_size)
    float* output_gradient = new float[layer3->output_size * batch_size];
    for (int i = 0; i < layer3->output_size * batch_size; i++) {
        output_gradient[i] = (layer3->last_output[i] - target[i]) / batch_size;
    }

    float* grad2 = new float[layer2->output_size * batch_size]; // gradient cho layer2 output
    float* grad1 = new float[layer1->output_size * batch_size]; // gradient cho layer1 output

    // Backward qua layer3
    layer3->backward(output_gradient, grad2, batch_size);

    // Backward qua layer2
    layer2->backward(grad2, grad1, batch_size);

    // Backward qua layer1
    layer1->backward(grad1, nullptr, batch_size); 
    // Layer1 là đầu vào, không cần input_gradient trả về layer trước

    delete[] output_gradient;
    delete[] grad2;
    delete[] grad1;
}

void ANN::update_weights(int batch_size) {
    // Cập nhật trọng số và độ chệch sau khi đã có gradient (tính trung bình theo batch)
    layer1->update_weights(learning_rate, batch_size);
    layer2->update_weights(learning_rate, batch_size);
    layer3->update_weights(learning_rate, batch_size);
}

void ANN::train(float* train_input, unsigned char* train_labels, int num_samples, int epochs, int batch_size) {
    int steps_per_epoch = num_samples / batch_size;
    std::cout << "[TRAIN] Starting training..." << std::endl;
    std::cout << "[TRAIN] Num samples: " << num_samples << ", Batch size: " << batch_size << ", Steps per epoch: " << steps_per_epoch << ", Epochs: " << epochs << std::endl;

    // Batch memory cho output và target
    float* batch_output = new float[layer3->output_size * batch_size];
    float* batch_target = new float[layer3->output_size * batch_size];

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int correct_count = 0;

        std::cout << "[TRAIN] Epoch " << (epoch+1) << " started." << std::endl;

        for (int step = 0; step < steps_per_epoch; step++) {
            float* input_batch = &train_input[step * batch_size * layer1->input_size];
            unsigned char* label_batch = &train_labels[step * batch_size];

            // Forward cả batch
            forward(input_batch, batch_output, batch_size);

            // Tạo batch_target (one-hot cho mỗi mẫu)
            for (int b = 0; b < batch_size; b++) {
                for (int i = 0; i < layer3->output_size; i++) {
                    batch_target[b * layer3->output_size + i] = 0.0f;
                }
                batch_target[b * layer3->output_size + label_batch[b]] = 1.0f;
            }

            // Tính loss cho batch (trung bình)
            float loss = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                for (int i = 0; i < layer3->output_size; i++) {
                    float pred = batch_output[b * layer3->output_size + i];
                    float tgt = batch_target[b * layer3->output_size + i];
                    if (tgt > 0.0f) {
                        loss -= logf(pred + 1e-9f);
                    }
                }
            }
            loss /= batch_size;
            epoch_loss += loss;

            // Tính accuracy trên batch
            for (int b = 0; b < batch_size; b++) {
                int pred = (int)(std::distance(&batch_output[b * layer3->output_size], 
                                               std::max_element(&batch_output[b * layer3->output_size], &batch_output[b * layer3->output_size + layer3->output_size])));
                if (pred == (int)label_batch[b]) correct_count++;
            }

            // Log chi tiết cho step
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
        std::cout << "[TRAIN] Epoch " << (epoch+1) << " completed. Avg Loss = " << avg_loss << ", Accuracy = " << accuracy << "%" << std::endl;
    }

    delete[] batch_output;
    delete[] batch_target;

    std::cout << "[TRAIN] Training finished." << std::endl;
}

void ANN::eval(float* test_input, unsigned char* test_labels, int test_size) {
    int correct_predictions = 0;
    // Eval từng mẫu hoặc theo batch (ở đây giữ đơn giản, mỗi mẫu)
    for (int i = 0; i < test_size; i++) {
        float* output = new float[layer3->output_size];
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
