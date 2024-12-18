#include <iostream>
#include <vector>
#include "loader.h"
#include "ANN.h"  // Bao gồm lớp ANN

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

    // Khởi tạo ANN model
    int input_size = 784;  // 28x28 pixels
    int hidden_size = 128;  // Số nơ-ron trong lớp ẩn
    int output_size = 10;   // 10 lớp đầu ra cho 10 nhãn (0-9)
    float learning_rate = 0.01f;

    // Các tham số huấn luyện
    int batch_size = 32;  // Kích thước của mỗi batch
    int num_epochs = 10;  // Số epochs (vòng lặp huấn luyện)

    // Tạo đối tượng ANN và huấn luyện mô hình
    ANN ann(input_size, hidden_size, output_size, learning_rate);
    
    // Chuyển dữ liệu huấn luyện vào ANN, với batch size và epoch
    ann.train(train_data.images.data(), train_data.labels.data(), batch_size, num_epochs);

    // Đánh giá mô hình trên bộ dữ liệu kiểm tra
    ann.eval(test_data.images.data(), test_data.labels.data(), test_data.num_samples);

    return 0;
}
