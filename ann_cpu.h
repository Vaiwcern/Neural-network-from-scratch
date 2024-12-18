// ann_cpu.h
#ifndef ANN_CPU_H
#define ANN_CPU_H

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <cassert>
#include <stdexcept>
#include <random>
#include "loader.h"

using namespace std;

// Activation Functions 
class ActivationFunction
{
public:
    virtual float activate(float x) const = 0;
    virtual float derivative(float x) const = 0;
    virtual ~ActivationFunction() {}
};

class ReLU : public ActivationFunction
{
public:
    float activate(float x) const override;
    float derivative(float x) const override;
};

class Softmax : public ActivationFunction
{
public:
    float activate(float x) const override;
    float derivative(float x) const override;

    vector<float> activate(const vector<float> &x) const;
};

// -------------------- Dense Layer -------------------- //
class DenseLayer
{
public:
    vector<vector<float>> weights; 
    vector<float> biases;          
    vector<float> inputs;          
    vector<float> Z;               
    vector<float> outputs;         
    int input_size, output_size;
    ActivationFunction *activation;

    // accumulators for gradients
    vector<vector<float>> gradW_acc;
    // accumulator for gradients of biases
    vector<float> gradB_acc;

    DenseLayer(int input_size, int output_size, ActivationFunction *activation);
    vector<float> forward(const vector<float> &input);
    vector<float> backward(const vector<float> &d_out, bool is_final_layer = false);
    void update_weights(float learning_rate, int batch_size);
    ~DenseLayer();
};

// -------------------- AnnModel -------------------- //
class AnnModel
{
public:
    DenseLayer layer1;
    DenseLayer layer2;
    DenseLayer layer3; // Output layer with softmax

    AnnModel();

    vector<float> forward(const vector<float> &input);
    float compute_loss(const vector<float> &pred, int label);
    vector<float> compute_output_gradient(const vector<float> &pred, int label);
    void backward(const vector<float> &d_out);
    void update_weights(float learning_rate, int batch_size);
    float inference(const Dataset &ds, int num_samples = -1);

    ~AnnModel() {}
};

// -------------------- Training Procedure -------------------- //
void train_model(AnnModel &model, const Dataset &ds,
                int num_epochs, int batch_size, float learning_rate);

#endif // ANN_CPU_H
