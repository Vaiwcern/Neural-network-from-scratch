#include <iostream>
#include <vector>
#include "loader.h"
#include "ANN.h"  // Bao gồm lớp ANN
#include "utils.h"

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

    // Chuẩn hóa dữ liệu hình ảnh cho huấn luyện
    float* train_images = new float[train_data.num_samples * train_data.image_size];
    normalize_data(train_data.images, train_images, train_data.num_samples, train_data.image_size);

    // Chuẩn hóa dữ liệu hình ảnh cho kiểm tra
    float* test_images = new float[test_data.num_samples * test_data.image_size];
    normalize_data(test_data.images, test_images, test_data.num_samples, test_data.image_size);

    // Chuẩn hóa nhãn cho huấn luyện
    float* train_labels = new float[train_data.num_samples * 10];  // One-hot encoding
    normalize_labels(train_data.labels, train_labels, train_data.num_samples);

    // Chuẩn hóa nhãn cho kiểm tra
    float* test_labels = new float[test_data.num_samples * 10];  // One-hot encoding
    normalize_labels(test_data.labels, test_labels, test_data.num_samples);


    train_images[0] = 0.1f;
    train_images[1] = 0.2f;
    train_images[2] = 0.3f;
    train_images[3] = 0.4f;
    train_images[4] = 0.5f;

    train_labels[0] = 0.0f;
    train_labels[0] = 1.0f;
    train_labels[0] = 0.0f;

    // Khởi tạo ANN model
    int input_size = 5;  // 28x28 pixels
    int hidden_size = 3;  // Số nơ-ron trong lớp ẩn
    int output_size = 2;   // 10 lớp đầu ra cho 10 nhãn (0-9)
    float learning_rate = 0.001f;

    // Các tham số huấn luyện
    int batch_size = 1;  // Kích thước của mỗi batch
    int num_epochs = 1;  // Số epochs (vòng lặp huấn luyện)

    // Tạo đối tượng ANN và huấn luyện mô hình
    ANN ann(input_size, hidden_size, output_size, learning_rate);
    
    // Huấn luyện mô hình với dữ liệu huấn luyện
    ann.train(train_images, train_labels, train_data.num_samples, batch_size, num_epochs);

    // // Đánh giá mô hình trên bộ dữ liệu kiểm tra
    // ann.eval(test_images, test_labels, test_data.num_samples);

    // Giải phóng bộ nhớ
    delete[] train_images;
    delete[] test_images;
    delete[] train_labels;
    delete[] test_labels;

    return 0;
}
