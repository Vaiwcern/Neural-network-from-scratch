#ifndef LOADER_H
#define LOADER_H

#include <string>
#include <vector>
#include <fstream>
#include <cuda_fp16.h>  // Thư viện hỗ trợ kiểu dữ liệu half

using namespace std;

// Cấu trúc Dataset với kiểu half
struct Dataset
{
    vector<half> images;  // Chuyển từ unsigned char sang half
    vector<unsigned char> labels;  // Vẫn giữ labels là unsigned char
    int num_samples;
    int image_size;
};

// Các phương thức load_data và readIDXFile với kiểu half
Dataset load_data(const string &image_file_train, const string &label_file_train);
vector<half> readIDXFile(const string &filename);  // Sử dụng half cho dữ liệu ảnh

#endif // LOADER_H
