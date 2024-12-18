#include <iostream>
#include <vector>  // Đảm bảo bạn bao gồm thư viện vector

using namespace std;  

// Hàm để chuẩn hóa dữ liệu hình ảnh từ unsigned char sang float
void normalize_data(vector<unsigned char>& images, float* output, int num_samples, int image_size) {
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < image_size; ++j) {
            // Chuyển đổi từ unsigned char (0-255) sang float (0.0 - 1.0)
            output[i * image_size + j] = images[i * image_size + j] / 255.0f;
        }
    }
}

void normalize_labels(vector<unsigned char>& labels, float* output, int num_samples) {
    for (int i = 0; i < num_samples; ++i) {
        int label = labels[i];
        for (int j = 0; j < 10; ++j) {
            output[i * 10 + j] = (j == label) ? 1.0f : 0.0f;
        }
    }
}
