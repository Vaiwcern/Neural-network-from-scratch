#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <cassert>

using namespace std;

class ActivationFunction {
public:
    virtual float activate(float x) const = 0;
    virtual vector<float> activate(const vector<float> &x) const {
        vector<float> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = activate(x[i]);
        }
        return result;
    }
};

class ReLU : public ActivationFunction {
public:
    float activate(float x) const override {
        return x > 0 ? x : 0;
    }
};

class Softmax : public ActivationFunction {
public:
    float activate(float x) const override {
        // Not used
        return x; 
    }

    vector<float> activate(const vector<float> &x) const override {
        vector<float> result(x.size());
        float max_val = *max_element(x.begin(), x.end());
        float sum_exp = 0.0f;

        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = exp(x[i] - max_val);
            sum_exp += result[i];
        }

        for (size_t i = 0; i < x.size(); ++i) {
            result[i] /= sum_exp;
        }

        return result;
    }
};

class DenseLayer {
public:
    vector<vector<float>> weights;
    vector<float> biases;
    int input_size, output_size;
    ActivationFunction *activation;

    DenseLayer(int input_size, int output_size, ActivationFunction *activation) 
        : input_size(input_size), output_size(output_size), activation(activation) 
    {
        weights.resize(output_size, vector<float>(input_size));
        biases.resize(output_size);
        
        srand(time(0));
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                weights[i][j] = ((float)rand() / RAND_MAX) * 2 - 1; 
            }
            biases[i] = ((float)rand() / RAND_MAX) * 2 - 1; 
        }
    }

    vector<float> forward(const vector<float> &input) {
        vector<float> output(output_size, 0.0f);
        
        for (int i = 0; i < output_size; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < input_size; ++j) {
                sum += weights[i][j] * input[j];
            }
            sum += biases[i];
            output[i] = sum;
        }

        return activation->activate(output);
    }
};


class ANN {
public:
    DenseLayer layer1;
    DenseLayer layer2;
    DenseLayer layer3;

    ANN() 
        : layer1(784, 128, new ReLU()), 
          layer2(128, 128, new ReLU()), 
          layer3(128, 10, new Softmax()) 
    {}

    vector<float> forward(const vector<float> &input) {
        vector<float> x = layer1.forward(input);
        x = layer2.forward(x);
        x = layer3.forward(x);
        return x;
    }
};


vector<float> generate_random_input() {
    vector<float> input(28 * 28);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = ((float)rand() / RAND_MAX); 
    }
    return input;
}

vector<unsigned char> readIDXFile(const string &filename)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        throw runtime_error("Cannot open file: " + filename);
    }

    int magic_number = 0;
    int num_items = 0;
    int rows = 0, cols = 0;

    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = __builtin_bswap32(magic_number);

    file.read((char *)&num_items, sizeof(num_items));
    num_items = __builtin_bswap32(num_items);

    if (magic_number == 0x00000803)
    {
        file.read((char *)&rows, sizeof(rows));
        file.read((char *)&cols, sizeof(cols));
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);
        cout << "Loaded " << num_items << " images with size " << rows << "x" << cols << endl;
    }
    else if (magic_number == 0x00000801)
    { 
        cout << "Loaded " << num_items << " labels" << endl;
    }
    else
    {
        throw runtime_error("Unknown magic number: " + to_string(magic_number));
    }

    vector<unsigned char> data((istreambuf_iterator<char>(file)), {});
    file.close();
    return data;
}


int main() {
    try {
        string image_file_train = "data/train-images-idx3-ubyte";
        string label_file_train = "data/train-labels-idx1-ubyte";

        vector<unsigned char> image_data_train = readIDXFile(image_file_train);
        vector<unsigned char> label_data_train = readIDXFile(label_file_train);

        int image_size = 28 * 28;

        cout << "\nFirst image" << endl;
        for (int i = 0; i < image_size; ++i) {
            if (i % 28 == 0)
                cout << endl;
            unsigned char pixel = image_data_train[16 + i];
            cout << (pixel > 128 ? '#' : '.') << " ";
        }
        
        cout << "\n\nLabel of first image: " << (int)label_data_train[8] << endl;
    } 
    catch (const exception &e) {
        cerr << "Error: " << e.what() << endl;
    }

    return 0;
}