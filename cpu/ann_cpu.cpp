// ann_cpu.cpp
#include "ann_cpu.h"

// Activation Functions Implementation
float ReLU::activate(float x) const
{
    return x > 0.0f ? x : 0.0f;
}

float ReLU::derivative(float x) const
{
    return x > 0.0f ? 1.0f : 0.0f;
}

float Softmax::activate(float x) const
{
    return x; // Not used individually
}

float Softmax::derivative(float x) const
{
    return 1.0f;
}

vector<float> Softmax::activate(const vector<float> &x) const
{
    vector<float> result(x.size());
    float max_val = *max_element(x.begin(), x.end());
    float sum_exp = 0.0f;

    for (size_t i = 0; i < x.size(); ++i)
    {
        result[i] = exp(x[i] - max_val);
        sum_exp += result[i];
    }

    for (size_t i = 0; i < x.size(); ++i)
    {
        result[i] /= sum_exp;
    }

    return result;
}

// DenseLayer Implementation
DenseLayer::DenseLayer(int input_size, int output_size, ActivationFunction *activation)
    : input_size(input_size), output_size(output_size), activation(activation)
{
    weights.resize(output_size, vector<float>(input_size));
    biases.resize(output_size);

    std::random_device rd;

    // pseudo-random generator
    std::mt19937 gen(rd());

    // He initialization (normal distribution)
    std::normal_distribution<float> dist(0.0f, sqrt(2.0f / input_size));

    for (int i = 0; i < output_size; ++i)
    {
        for (int j = 0; j < input_size; ++j)
        {
            weights[i][j] = dist(gen);
        }
        biases[i] = 0.0f;
    }

    gradW_acc.resize(output_size, vector<float>(input_size, 0.0f));
    gradB_acc.resize(output_size, 0.0f);
}

vector<float> DenseLayer::forward(const vector<float> &input)
{
    inputs = input;
    Z.resize(output_size, 0.0f);

    // compute the weighted sum of inputs
    for (int i = 0; i < output_size; ++i)
    {
        float sum = biases[i];
        for (int j = 0; j < input_size; ++j)
        {
            sum += weights[i][j] * input[j];
        }
        Z[i] = sum;
    }

    // because the activation function of Softmax is different
    Softmax *softmax_ptr = dynamic_cast<Softmax *>(activation);
    if (softmax_ptr)
    {
        outputs = softmax_ptr->activate(Z);
    }
    else
    {
        outputs.resize(output_size);
        for (int i = 0; i < output_size; i++)
        {
            outputs[i] = activation->activate(Z[i]);
        }
    }

    return outputs;
}

vector<float> DenseLayer::backward(const vector<float> &d_out, bool is_final_layer)
{
    vector<float> dZ = d_out;

    // because the activation function of Softmax is different
    if (!is_final_layer)
    {
        for (size_t i = 0; i < Z.size(); ++i)
        {
            float dAct = activation->derivative(Z[i]);
            dZ[i] = d_out[i] * dAct;
        }
    }


    for (int i = 0; i < output_size; ++i)
    {
        for (int j = 0; j < input_size; ++j)
        {
            // compute gradients
            gradW_acc[i][j] += dZ[i] * inputs[j];
        }

        // compute gradients of biases
        gradB_acc[i] += dZ[i];
    }

    vector<float> dX(input_size, 0.0f);
    for (int j = 0; j < input_size; ++j)
    {
        float sum = 0.0f;
        for (int i = 0; i < output_size; ++i)
        {
            sum += weights[i][j] * dZ[i];
        }

        dX[j] = sum;
    }

    // return the gradient for the backpropagation
    return dX;
}

void DenseLayer::update_weights(float learning_rate, int batch_size)
{
    for (int i = 0; i < output_size; ++i)
    {
        for (int j = 0; j < input_size; ++j)
        {
            // update weights
            weights[i][j] -= (learning_rate / batch_size) * gradW_acc[i][j];
        }

        // update biases
        biases[i] -= (learning_rate / batch_size) * gradB_acc[i];
    }

    // reset accumulators
    for (int i = 0; i < output_size; ++i)
    {
        fill(gradW_acc[i].begin(), gradW_acc[i].end(), 0.0f);
        gradB_acc[i] = 0.0f;
    }
}

DenseLayer::~DenseLayer()
{
    delete activation;
}

// AnnModel Implementation
AnnModel::AnnModel()
    : layer1(784, 128, new ReLU()),
      layer2(128, 128, new ReLU()),
      layer3(128, 10, new Softmax())
{}

vector<float> AnnModel::forward(const vector<float> &input)
{
    auto x = layer1.forward(input);
    x = layer2.forward(x);
    x = layer3.forward(x);
    return x;
}

float AnnModel::compute_loss(const vector<float> &pred, int label)
{
    float epsilon = 1e-8f;

    return -log(pred[label] + epsilon);
}

vector<float> AnnModel::compute_output_gradient(const vector<float> &pred, int label)
{
    vector<float> grad(pred.size());

    // cross-entropy loss with softmax
    for (size_t i = 0; i < pred.size(); ++i)
    {
        grad[i] = pred[i];
    }
    grad[label] -= 1.0f;
    return grad;
}

void AnnModel::backward(const vector<float> &d_out)
{
    auto dX2 = layer3.backward(d_out, true);
    auto dX1 = layer2.backward(dX2, false);
    layer1.backward(dX1, false);
}

void AnnModel::update_weights(float learning_rate, int batch_size)
{
    layer1.update_weights(learning_rate, batch_size);
    layer2.update_weights(learning_rate, batch_size);
    layer3.update_weights(learning_rate, batch_size);
}

float AnnModel::inference(const Dataset &ds, int num_samples)
{
    if (num_samples < 0 || num_samples > ds.num_samples)
        num_samples = ds.num_samples;

    int correct = 0;
    int image_size = ds.image_size;

    for (int i = 0; i < num_samples; ++i)
    {
        vector<float> input(image_size);
        for (int p = 0; p < image_size; ++p)
        {
            unsigned char pixel = ds.images[i * image_size + p];
            input[p] = pixel / 255.0f;
        }

        int label = (int)ds.labels[i];
        vector<float> pred = forward(input);
        auto max_pos = max_element(pred.begin(), pred.end());
        int predicted_label = (int)distance(pred.begin(), max_pos);

        if (predicted_label == label)
            correct++;
    }

    return (float)correct / num_samples;
}

// Training Procedure Implementation 
void train_model(AnnModel &model, const Dataset &ds,
                 int num_epochs, int batch_size, float learning_rate)
{
    int num_samples = ds.num_samples;
    int image_size = ds.image_size;

    vector<int> indices(num_samples);
    iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());

    // Normalize pixel values to [0, 1] and compute initial loss
    {
        float initial_loss = 0.0f;
        int check_samples = 100;
        for (int i = 0; i < check_samples; ++i)
        {
            vector<float> input(image_size);
            for (int p = 0; p < image_size; ++p)
            {
                unsigned char pixel = ds.images[i * image_size + p];
                input[p] = pixel / 255.0f;
            }
            int label = (int)ds.labels[i];
            vector<float> pred = model.forward(input);
            initial_loss += model.compute_loss(pred, label);
        }
        cout << "Initial average loss (first 100 samples): " << (initial_loss / 100.0f) << endl;
    }

    // Training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {
        // Shuffle indices
        shuffle(indices.begin(), indices.end(), g);

        float total_loss = 0.0f;
        int count = 0;

        // loop over all samples in the dataset
        for (int i = 0; i < num_samples; i += batch_size)
        {
            int current_batch_size = min(batch_size, num_samples - i);
            float batch_loss = 0.0f;

            // loop over the current batch
            for (int b = 0; b < current_batch_size; ++b)
            {
                int idx = indices[i + b];
                vector<float> input(image_size);

                // normalize pixel values to [0, 1]
                for (int p = 0; p < image_size; ++p)
                {
                    unsigned char pixel = ds.images[idx * image_size + p];
                    input[p] = pixel / 255.0f;
                }

                int label = (int)ds.labels[idx];

                vector<float> pred = model.forward(input);
                float loss = model.compute_loss(pred, label);
                batch_loss += loss;

                vector<float> d_out = model.compute_output_gradient(pred, label);
                model.backward(d_out);
            }

            model.update_weights(learning_rate, current_batch_size);

            total_loss += batch_loss;
            count += current_batch_size;

            if ((i / batch_size) % 100 == 0)
            {
                cout << "[Epoch " << epoch << "][Batch " << (i / batch_size)
                     << "] Average batch loss: " << (batch_loss / current_batch_size) << endl;
            }
        }

        float avg_loss = total_loss / count;
        float train_accuracy = model.inference(ds, 10000);
        cout << "Epoch " << epoch << " finished with average loss: " << avg_loss
             << ", partial train accuracy (10k samples): " << train_accuracy << endl;
    }
}