#include <iostream>
#include <vector>
#include "loader.h"

using namespace std;

int main() {
    // Đường dẫn đến các tệp dataset
    string image_file_train = "./data/train-images-idx3-ubyte";
    string label_file_train = "./data/train-labels-idx1-ubyte";
    string image_file_test = "./data/t10k-images-idx3-ubyte";
    string label_file_test = "./data/t10k-labels-idx1-ubyte";

    // Load dữ liệu huấn luyện và kiểm tra
    Dataset train_data = load_data(image_file_train, label_file_train);
    Dataset test_data = load_data(image_file_test, label_file_test);

    // In thông tin về dữ liệu huấn luyện
    cout << "Training data:" << endl;
    cout << "Number of samples: " << train_data.num_samples << endl;
    cout << "Image size: " << train_data.image_size << " pixels" << endl;

    // In thông tin về dữ liệu kiểm tra
    cout << "\nTest data:" << endl;
    cout << "Number of samples: " << test_data.num_samples << endl;
    cout << "Image size: " << test_data.image_size << " pixels" << endl;

    // In nhãn của một số mẫu để kiểm tra
    cout << "\nSample labels from training data:" << endl;
    for (int i = 0; i < 10; ++i) {
        cout << "Label for image " << i + 1 << ": " << (int)train_data.labels[i] << endl;
    }

    // In vài pixel đầu tiên của một số hình ảnh để kiểm tra
    cout << "\nSample image pixels (first 10 pixels) from the first image in training data:" << endl;
    for (int i = 0; i < 10; ++i) {
        cout << (int)train_data.images[i] << " ";
    }
    cout << endl;

    return 0;
}
