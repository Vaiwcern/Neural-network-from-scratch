#include <iostream>
#include <vector>
#include "ANN.h"
#include "loader.h"
#include "Macro.h"
#include <cuda_fp16.h>  // Thêm thư viện để làm việc với half precision

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
    vector<half> train_images_half(train_data.images.size());
    for (size_t i = 0; i < train_data.images.size(); i++) {
        train_images_half[i] = __float2half((float)train_data.images[i] / 255.0f);
    }

    vector<half> test_images_half(test_data.images.size());
    for (size_t i = 0; i < test_data.images.size(); i++) {
        test_images_half[i] = __float2half((float)test_data.images[i] / 255.0f);
    }

    // Chuyển đổi nhãn từ unsigned char về float16 (half)
    vector<unsigned char> train_labels = train_data.labels;
    vector<unsigned char> test_labels = test_data.labels;

    // Khởi tạo mô hình ANN
    int input_size = 28*28;
    int hidden_size = 128;
    int output_size = 10;
    float learning_rate = 0.01f;

    ANN net(input_size, hidden_size, output_size, learning_rate);

    int epochs = 10;      
    int batch_size = 32; 

    GpuTimer timer;
    timer.Start();
    cout << "Start Training..." << endl;
    net.train(train_images_half.data(), train_labels.data(), 60000, epochs, batch_size);  // Gọi hàm train với kiểu half
    timer.Stop();

    cout << "Training time: " << timer.Elapsed() << " ms" << endl;

    cout << "Evaluate on Test set..." << endl;
    net.eval(test_images_half.data(), test_labels.data(), test_data.num_samples);  // Gọi hàm eval với kiểu half

    return 0;
}
