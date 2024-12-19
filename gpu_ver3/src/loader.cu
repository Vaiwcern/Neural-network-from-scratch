#include "loader.h"
#include <stdexcept>
#include <iostream>
#include <cuda_fp16.h>  // Thư viện hỗ trợ kiểu dữ liệu half

static int32_t readInt(ifstream &f)
{
    int32_t val = 0;
    f.read((char *)&val, 4);
    // convert from big-endian to little-endian
    val = ((val & 0x000000FF) << 24) |
          ((val & 0x0000FF00) << 8) |
          ((val & 0x00FF0000) >> 8) |
          ((val & 0xFF000000) >> 24);
    return val;
}

vector<half> readIDXFile(const string &filename)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        throw runtime_error("Cannot open file: " + filename);
    }

    int magic_number = readInt(file);
    int num_items = readInt(file);

    // nếu là ảnh
    if (magic_number == 0x00000803)
    {
        int rows = readInt(file);
        int cols = readInt(file);
        vector<half> data(num_items * rows * cols);
        vector<unsigned char> raw_data(num_items * rows * cols);

        // Đọc dữ liệu vào mảng raw_data
        file.read((char *)raw_data.data(), raw_data.size());

        // Chuyển dữ liệu từ unsigned char sang half
        for (size_t i = 0; i < raw_data.size(); ++i) {
            data[i] = __float2half(static_cast<float>(raw_data[i]) / 255.0f);  // Chuyển từ unsigned char (0-255) thành half (0.0f - 1.0f)
        }

        return data;
    }
    else if (magic_number == 0x00000801)
    {
        // labels
        vector<half> data(num_items);
        vector<unsigned char> raw_data(num_items);
        file.read((char *)raw_data.data(), raw_data.size());

        // Chuyển từ unsigned char (0-255) sang half
        for (size_t i = 0; i < raw_data.size(); ++i) {
            data[i] = __float2half(static_cast<float>(raw_data[i]) / 255.0f);  // Normalizing label to [0.0, 1.0]
        }

        return data;
    }
    else {
        throw runtime_error("Invalid IDX file: " + filename);
    }
}

Dataset load_data(const string &image_file_train, const string &label_file_train)
{
    Dataset ds;
    ds.images = readIDXFile(image_file_train);
    ds.labels = readIDXFile(label_file_train);
    ds.num_samples = (int)ds.labels.size();
    ds.image_size = 28 * 28;
    return ds;
}
