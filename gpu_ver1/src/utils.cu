// Hàm để chuẩn hóa dữ liệu hình ảnh từ unsigned char sang float
void normalize_data(vector<unsigned char>& images, float* output, int num_samples, int image_size) {
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < image_size; ++j) {
            // Chuyển đổi từ unsigned char (0-255) sang float (0.0 - 1.0)
            output[i * image_size + j] = images[i * image_size + j] / 255.0f;
        }
    }
}
