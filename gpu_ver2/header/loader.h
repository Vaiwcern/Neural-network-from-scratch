#ifndef LOADER_H
#define LOADER_H

#include <string>
#include <vector>
#include <fstream>

using namespace std;

struct Dataset
{
    vector<unsigned char> images;
    vector<unsigned char> labels;
    int num_samples;
    int image_size;
};

Dataset load_data(const string &image_file_train, const string &label_file_train);
vector<unsigned char> readIDXFile(const string &filename);

#endif // LOADER_H
