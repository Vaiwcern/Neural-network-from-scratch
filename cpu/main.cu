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
#include <cuda.h>

#include "loader.h"
#include "ann_cpu.h"

using namespace std;

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

int main()
{
    try
    {
        string image_file_train = "../data/train-images-idx3-ubyte";
        string label_file_train = "../data/train-labels-idx1-ubyte";
        string image_file_test = "../data/t10k-images-idx3-ubyte";
        string label_file_test = "../data/t10k-labels-idx1-ubyte";

        Dataset ds_train = load_data(image_file_train, label_file_train);
        Dataset ds_test = load_data(image_file_test, label_file_test);

        AnnModel model;
        int batch_size = 32;
        float learning_rate = 0.01f;
        int num_epochs = 10;

        cout << "Starting training..." << endl;
        GpuTimer timer;
        timer.Start();
        train_model(model, ds_train, num_epochs, batch_size, learning_rate);
        timer.Stop();

        cout << "Training time: " << timer.Elapsed() << " ms" << endl;

        float final_accuracy = model.inference(ds_test, 10000);
        cout << "Accuracy on test dataset" << final_accuracy << endl;
    }
    catch (const exception &e)
    {
        cerr << "Error: " << e.what() << endl;
    }

    return 0;
}