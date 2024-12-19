#include <iostream>
#include <vector>
#include "ANN.h"
#include "loader.h"
#include "Macro.h"

using namespace std;

int main() {
    string image_file_train = "../data/train-images-idx3-ubyte";
    string label_file_train = "../data/train-labels-idx1-ubyte";
    string image_file_test = "../data/t10k-images-idx3-ubyte";
    string label_file_test = "../data/t10k-labels-idx1-ubyte";

    cout << "Loading data..." << endl;
    Dataset train_data = load_data(image_file_train, label_file_train);
    Dataset test_data = load_data(image_file_test, label_file_test);

    // Chuyển đổi ảnh từ unsigned char [0..255] về float [0..1]
    vector<float> train_images_float(train_data.images.size());
    for (size_t i = 0; i < train_data.images.size(); i++) {
        train_images_float[i] = (float)train_data.images[i] / 255.0f;
    }

    vector<float> test_images_float(test_data.images.size());
    for (size_t i = 0; i < test_data.images.size(); i++) {
        test_images_float[i] = (float)test_data.images[i] / 255.0f;
    }

    // Khởi tạo mô hình ANN
    int input_size = 28*28;
    int hidden_size = 128;
    int output_size = 10;
    float learning_rate = 0.01f;

    ANN net(input_size, hidden_size, output_size, learning_rate);

    int epochs = 1;      
    int batch_size = 32; 

    GpuTimer timer;
    timer.Start();
    cout << "Start Training..." << endl;
    net.train(train_images_float.data(), train_data.labels.data(), 60000, epochs, batch_size);
    timer.Stop();

    cout << "Training time: " << timer.Elapsed() << " ms" << endl;

    cout << "Evaluate on Test set..." << endl;
    net.eval(test_images_float.data(), test_data.labels.data(), test_data.num_samples);

    return 0;
}