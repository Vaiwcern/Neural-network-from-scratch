#include "loader.h"

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

vector<unsigned char> readIDXFile(const string &filename)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        throw runtime_error("Cannot open file: " + filename);
    }

    int magic_number = readInt(file);
    int num_items = readInt(file);

    // skip dimensions for images
    if (magic_number == 0x00000803)
    {
        readInt(file);
        readInt(file);
    }

    return vector<unsigned char>((istreambuf_iterator<char>(file)), {});
}

Dataset load_data(const string &image_file_train, const string &label_file_train)
{
    Dataset ds;
    ds.images = readIDXFile(image_file_train);
    ds.labels = readIDXFile(label_file_train);
    ds.num_samples = ds.labels.size();
    ds.image_size = 28 * 28;
    return ds;
}
