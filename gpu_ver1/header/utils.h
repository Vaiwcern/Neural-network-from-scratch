#pragma once 

// Hàm để chuẩn hóa dữ liệu hình ảnh từ unsigned char sang float
void normalize_data(vector<unsigned char>& images, float* output, int num_samples, int image_size);